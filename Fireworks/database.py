from sqlalchemy import Table, Column, Integer
from sqlalchemy.ext.declarative import declarative_base
from Fireworks.datasource import Source


Base = declarative_base()

class TableSource(Source):
    """
    Represents an SQLalchemy Table while having the functionality of a Source.
    """
    def __init__(self, input_sources, table, **kwargs):
        super().__init__(input_sources, **kwargs)
        self.table = table # An SQLalchemy table class

    def reset(self): pass

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

    def upsert(self, batch): pass

    def make_row(self, row):

        kwargs = {key: row[key] for key in self.columns}
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


# Get Representation
# Test row generation
# Test with SQLite backend
# Test with CSV, Message, and Fasta
# Create reader sources
