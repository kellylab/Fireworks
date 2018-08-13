from sqlalchemy import Table, Column, Base

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

def create_table(name, columns_dict):
    """
    Creates a table given a dict of column names to data types. This is an easy
    way to quickly create a schema for a data pipeline.
    """

    class SimpleTable(Base):
        __tablename__ = name
        pass 
