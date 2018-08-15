from sqlalchemy import Table, Column, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from Fireworks import Message
from Fireworks.datasource import Source, PassThroughSource


Base = declarative_base()

class TableSource(PassThroughSource):
    """
    Represents an SQLalchemy Table while having the functionality of a Source.
    """
    def __init__(self, table, engine, columns = None, inputs = None, **kwargs):
        super().__init__(inputs=inputs, **kwargs)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.table = table # An SQLalchemy table class
        self.columns = columns or parse_columns(table)

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()

    def insert(self, batch):
        """
        Inserts the contents of batch message into the database using self.table object
        """

        rows = [self.make_row(row) for row in batch]
        self.session.add_all(rows)

    def query(self, entities=None, *args, **kwargs):

        if entities is None:
            entities = self.table

        return self.session.query(entities, *args, **kwargs)

    def upsert(self, batch): pass

    def make_row(self, row):

        kwargs = {key: row[key][0] for key in self.columns}
        return self.table(**kwargs)

def create_table(name, columns, primary_key = None):
    """
    Creates a table given a dict of column names to data types. This is an easy
    way to quickly create a schema for a data pipeline.
    """

    if primary_key is None: # Create a default, autoincrementing primary key
        columns.insert(0, Column('id', Integer, primary_key=True, autoincrement=True)) # Prepend to columns list

    class SimpleTable(Base):
        __table__ = Table(name, Base.metadata, *columns)

        def __repr__(self):
            return "Table {name} with values {values}".format(name=self.__table__.name,
            values = ", ".join(["{0}={1}".format(c.name, self.__getattribute__(c.name)) for c in columns]))

    return SimpleTable

class DBSource(Source):
    """
    Source that can iterate through the output of a database query.
    """
    def __init__(self, table, engine, query = None):
        Session = sessionmaker(bind=engine)
        self.session = Session()
        self.table = table
        self.columns_and_types = parse_columns_and_types(table)

    def __iter__(self):

        self.reset()
        self.iterator = self.query.__iter__()
        return self

    def reset(self, entities=None, *args, **kwargs):

        if entities is None:
            entities = self.table

        self.query = self.session.query(entities, *args, **kwargs)

    def __next__(self):

        return to_message(self.iterator.__next__(), columns_and_types=self.columns_and_types)

def parse_columns(table):
    """
    Returns column names in a table object
    """
    return [c.key for c in table.__table__.columns]

def parse_columns_and_types(table):
    """
    Returns column names and types in a table object as a dict
    """
    return {c.key: c.type for c in table.__table__.columns}

def convert(value, sqltype):

    objtype = type(sqltype)
    if objtype is Integer:
        if type(value) is bytes:
            value = int.from_bytes(value, byteorder='little')
        return int(value)

    return value

def to_message(row, columns_and_types=None):

    if columns_and_types is None:
        columns_and_types = parse_columns_and_types(row)

    row_dict = {c: [convert(getattr(row,c), t)] for c,t in columns_and_types.items()}

    return Message(row_dict)

# Get Representation
# Test row generation
# Test with SQLite backend
# Test with CSV, Message, and Fasta
# Create reader sources
