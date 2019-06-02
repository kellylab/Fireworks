import sqlalchemy
from sqlalchemy import Table, Column, Integer, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Query
from fireworks.core import Message, cat, Pipe
import numpy as np
import pandas as pd

#TODO: Develop a way to infer schemas from Messages and Pipes to make save/load easier.

class TablePipe(Pipe):
    """
    Represents an SQLalchemy Table while having the functionality of a Pipe.
    """
    def __init__(self, table, engine, columns = None, input = None, **kwargs):

        super().__init__(input=input, **kwargs)
        self.sessionmaker = sessionmaker(bind=engine)
        self.reset_session()
        self.engine = engine
        if type(table) is str:
            table = reflect_table(table, engine)
        self.table = table # An SQLalchemy table class
        self.columns = columns or parse_columns(table)
        self.init_db()

    def init_db(self):
        """
        Initializes metadata for internal table. This ensures that the table exists in the database.
        """
        self.table.metadata.create_all(self.engine)

    def commit(self):
        """
        Commits transactions to database.
        """
        self.session.commit()

    def rollback(self):
        """
        Rollbacks transactions in current session that haven't been committed yet.
        """
        self.session.rollback()

    def reset_session(self, session=None):

        try:
            self.session.close()
        except:
            pass
        self.session = session or self.sessionmaker()

    def insert(self, batch):
        """
        Inserts the contents of batch message into the database using self.table object
        NOTE: Only the dataframe components of a message will be inserted.

        Args:
            batch (Message): A message to be inserted. The columns and types must be consistent with the database schema.
        """

        # TODO:     Enable inserting TensorMessages (QUESTION: how would this work?)
        rows = list(batch.df.T.to_dict().values())

        # NOTE:     Bulk inserts implemented using this guide:
        #           http://docs.sqlalchemy.org/en/latest/faq/performance.html#i-m-inserting-400-000-rows-with-the-orm-and-it-s-really-slow

        # QUESTION: Should this be the standard insert method, or should this be configurable?
        #           There are tradeoffs between doing bulk inserts and the default add method.
        #           See here for more details: http://docs.sqlalchemy.org/en/latest/orm/persistence_techniques.html#bulk-operations

        if hasattr(self.table, '__table__'):
            table = self.table.__table__
        elif type(self.table) is Table:
            table = self.table
        else:
            raise ValueError("table must either be an SQLalchemy table or have an __table__ attribute that is.")

        self.session.execute(
            table.insert(), # NOTE: This requires self.table to have a __table__ attribute, which is not guaranteed.
            rows
        )

    def query(self, entities=None, *args, **kwargs):
        """
        Queries the database and generates a DBPipe corresponding to the result.

        Args:
            entities: A list of column names
            args: Optional positional arguments for the SQLalchemy query function
            kwargs: Optional keyword arguments for the SQLalchemy query function

        Returns:
            dbpipe (DBPipe): A DBPipe object that can iterate through the results of the query.
        """
        if entities is None:
            query = self.session.query(*args, **kwargs)
            if query.column_descriptions == []:
                columns_and_types = None
            else:
                columns_and_types = parse_columns_and_types(self.table)
            # columns_and_types = None
        elif type(entities) is str:
            entities = getattr(self.table, entities)
            query = self.session.query(entities, *args, **kwargs)
            columns_and_types = parse_columns_and_types(query)
        elif type(entities) is list:
            entities = [getattr(self.table, entity) for entity in entities]
            query = self.session.query(*entities, *args, **kwargs)
            columns_and_types = parse_columns_and_types(query)

        return DBPipe(self.table, self.engine, query, columns_and_types = columns_and_types)

    def delete(self, column_name, values):
        to_delete = self.query().filter(column_name, 'in_', values)
        to_delete.delete()

    def update(self, filter_column, batch): # TODO: Support filters with multiple simultaneous columns
        to_update = self.query().filter(filter_column, 'in_', batch[filter_column])
        to_update.update(batch)

    def upsert(self, batch):
        """
        Performs an upsert into the database. This is equivalent to performing an update + insert (ie. if value is not present, insert it,
        otherwise update the existing value.)

        Args:
            batch (Message): The message to upsert.
        """
        pass


    def make_row(self, row):
        """
        Converts a Message or dict mapping columns to values into a table object that can be inserted into an SQLalchemy database.

        Args:
            row: row in Message or dict form to convert.

        Returns:
            table: row converted to table form.
        """

        kwargs = {key: cast(row[key][0]) for key in self.columns}

        return self.table(**kwargs)

    def make_row_dict(self, row):
        """
        Converts a 1-row Message into a dict of atomic (non-listlike) elements. This can be used for the bulk_insert_mappings method of
        an SQLalchemy session, which skips table instantiation and takes dictionaries as arguments instead.

        Args:
            row: row in Message or dict form to convert.

        Returns:
            table: row converted to table form.
        """

        kwargs = {key: cast(row[key][0]) for key in self.columns}

        return kwargs

def create_table(name, columns, primary_key = None, Base=None):
    """
    Creates a table given a dict of column names to data types. This is an easy
    way to quickly create a schema for a data pipeline.

    Args:
        columns (dict): Dict mapping column names to SQLalchemy types.
        primary_key: The column that should be the primary key of the table. If unspecified, a new auto-incrementing column called 'id'
            will be added as the primary key. SQLalchemy requires that all tables have a primary key, and this ensures that every row
            is always uniquely identifiable.
        Base : An optional argument that can be provided to specify the Base class that the new table class will inherit from. By
            default, this will be set to an instance of declarative_base from SQLalchemy.

    Returns:
        simpletable (sqlalchemy.ext.declarative.api.DeclarativeMeta): A table class specifying the schema for the database table.
    """
    if primary_key is None: # Create a default, autoincrementing primary key # NOTE: TODO: If a primary key is desired, we need to specify that
        columns.insert(0, Column('id', Integer, primary_key=True, autoincrement=True)) # Prepend to columns list

    Base = Base or declarative_base()

    class SimpleTable(Base):
        __table__ = Table(name, Base.metadata, *columns)

        def __repr__(self):
            return "Table {name} with values {values}".format(name=self.__table__.name,
            values = ", ".join(["{0}={1}".format(c.name, self.__getattribute__(c.name)) for c in columns]))

    return SimpleTable

class DBPipe(Pipe):
    """
    Pipe that can iterate through the output of a database query.
    """
    def __init__(self, table, engine, query = None, columns_and_types= None):
        """
        Args:
            table (sqlalchemy.ext.declarative.api.DeclarativeMeta): Table to perform query on. You can alternatively provide the name of the
                table as a string, and the schema will be extracted from the engine.
            engine (sqlalchemy.engine.base.Engine): Engine correspondign to the database to read from.
            query: Can optionally provide an SQLalchemy Query object. If unspecified, the DBPipe will perform a SELECT * query.
        """
        Pipe.__init__(self)
        self.engine = engine
        self.sessionmaker = sessionmaker(bind=engine)
        if query is None:
            self.reset_session()
            self.query = self.session.query()
        else:
            self.query = query # or self.session.query()
            self.reset_session(query.session)
        if type(table) is str:
            self.table = reflect_table(table, engine)
        else:
            self.table = table

        if columns_and_types is None:
            self.columns_and_types = parse_columns_and_types(self.query, ignore_id=False)
            if self.columns_and_types == {}: # Remap empty query to SELECT *
                self.columns_and_types = parse_columns_and_types(self.table, ignore_id=False)
                self.query = self.session.query(self.table)
        else:
            self.columns_and_types = columns_and_types

    def reset(self):

        self.__iter__()

    def __iter__(self):

        self.iterator = self.query.__iter__()
        return self

    def reset_session(self, session=None):

        if hasattr(self, 'session'):
            self.session.close()

        self.session = session or self.sessionmaker()

        if hasattr(self, 'query'):
            self.query.session = self.session

    def __next__(self):

        return to_message(self.iterator.__next__(), columns_and_types=self.columns_and_types)

    def __len__(self):

        return self.query.count()

    def filter(self, column_name, predicate, *args, **kwargs):
        """
        Applies an sqlalchemy filter to query.
        """
        column = getattr(self.table, column_name)
        predicate_function = getattr(column, predicate)
        query = self.query.filter(predicate_function(*args, **kwargs))
        filtered = DBPipe(self.table, self.engine, query, columns_and_types=self.columns_and_types)
        return filtered

    def all(self): #TODO: Test dis ish #TODO: Implement the other query methods
        """
        Returns the results of the query as a single Message object.
        """
        return cat([to_message(x, columns_and_types=self.columns_and_types) for x in self.query.all()])

    def delete(self):
        self.query.delete(synchronize_session=False)

    def update(self, batch):
        """
        Updates the contents of this DBPipe by replacing them with batch

        Args:
            batch: A Message
        """
        rows = list(batch.df.T.to_dict().values())
        self.session.bulk_update_mappings(self.table, rows)

def parse_columns(object, ignore_id=True):
    """
    Returns the names of columns in a table or query object

    Args:
        table (sqlalchemy.ext.declarative.api.DeclarativeMeta):
        ignore_id (bool): If True, ignore the 'id' column, which is a default primary key
            added by the create_table function.

    Returns:
        columns (list): A list of columns names in the sqlalchemy object.
    """

    return list(parse_columns_and_types(object, ignore_id).keys())

def parse_columns_and_types(object, ignore_id = True):
    """
    Returns column names and types in a object or query object as a dict

    Args:
        object: An SQLalchemy table or Query object
        ignore_id (bool): If True, ignore the 'id' column, which is a default primary key
            added by the create_table function.

    Returns:
        columns_and_types (dict): A dict mapping column names to their SQLalchemy type.
    """
    if type(object) is Query:
        columns_and_types = {col['name']: col['type'] for col in object.column_descriptions} #TODO: Test this
    elif hasattr(object, '__table__'):
        columns_and_types = {str(c.key): c.type for c in object.__table__.columns}
    elif type(object) is Table:
        columns_and_types = {str(c.key): c.type for c in object.columns}
    else:
        raise AttributeError("Could not extract column from table.")
    if ignore_id and 'id' in columns_and_types:
        del columns_and_types['id']
    return columns_and_types

def to_message(row, columns_and_types=None):
    """
    Converts a database query result produced by SQLalchemy into a Message

    Args:
        row: A row from the query.
        columns_and_types (dict): If unspecified, this will be inferred. Otherwise,
            you can specify the columns
            to parse, for example, if you only want to extract some columns.
    Returns:
        message: Message representation of input.
    """
    if columns_and_types is None:
        columns_and_types = parse_columns_and_types(row)

    row_dict = {c: [getattr(row,c)] for c in columns_and_types.keys()}

    return Message(row_dict)

def cast(value):
    """
    Converts values to basic types (ie. np.int64 to int)

    Args:
        value: The object to be cast.

    Returns:
        The cast object.

    """
    if type(value) is pd.Series:
        value = value[0]
    if type(value) is np.int64:
        value = int(value)
    if type(value) is np.float64:
        value = float(value)
    if type(value) is pd.Timestamp:
        value = value.to_pydatetime()

    return value

def reflect_table(table_name, engine):
    """
    Gets the table with the given name from the sqlalchemy engine.

    Args:
        table_name (str): Name of the table to extract.
        engine (sqlalchemy.engine.base.Engine): Engine to extract from.

    Returns:
        table (sqlalchemy.ext.declarative.api.DeclarativeMeta): The extracted table, which can be now be used to read from the database.
    """
    meta = MetaData()
    table = Table(table_name, meta, autoload=True, autoload_with=engine)
    return table
