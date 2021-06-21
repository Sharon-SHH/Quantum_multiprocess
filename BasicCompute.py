from datetime import datetime, timedelta

class BasicCompute(object):
    """Given a certain date, find and return the index of the date in df."""
    def __init__(self, df):
        self.df = df         # the selected dataframe

    def add_one_day(self, selected_date, days=1):
        to_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
        return (to_date + timedelta(days=days))

    def get_index(self, df, selected_date):
        ''' return index which equal to the date '''
        row = df.loc[df['x'] == selected_date]
        return row

    def get_meaningful_index(self, date):
        row = self.get_index(self.df, date)
        days = 1
        while row.size == 0:
            mydate = self.add_one_day(date, days)
            new_date = mydate.strftime('%Y-%m-%d')
            row = self.get_index(self.df, new_date)
            days += 1
        return row.index[0]
