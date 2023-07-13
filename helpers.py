import os.path as op

import dask.dataframe as dd
import pandas as pd

import config

from ast import literal_eval

# nice columns to keep (i.e., to reduce the data frame size)
KEEPERS = [
    'context_annotations',
    'created_at',
    'date',
    'in_reply_to_user_id',
    'lang',
    'likes',
    'media',
    'name',
    'quotes',
    'replies',
    'retweets',
    'text',
    'tweet_url',
    'tweet_count',
    'type',
    'user_bio',
    'username',
    'verified',
]


def format_df(df, incl_html=False):
    '''Format a Twitter dataframe in Pandas or Dask (including Parquet).'''
    str_meta = pd.Series(dtype='string')

    # if `df` is a Dask dataframe...
    try:

        # add 'tweet_url' column
        username = 'author.username' if 'author.username' in df else 'username'
        df['tweet_url'] = df[[username, 'id']].apply(
            make_tweet_url,
            axis=1,
            meta=str_meta,
        )

        # add 'media' column
        df['media'] = df['attachments.media'].apply(
            get_image_url,
            meta=str_meta,
        )

        # optionally render clickable media images
        if incl_html:
            df['media'] = df['media'].apply(
                get_image_html,
                meta=str_meta,
                )

        # add 'type' column (i.e., tweet type)
        df['type'] = df.apply(
            find_tweet_type,
            axis=1,
            meta=pd.Series(dtype='category'),
        )

    # if `df` is a Pandas dataframe...
    except TypeError:

        # add 'tweet_url' column
        df['tweet_url'] = df[[username, 'id']].apply(make_tweet_url, axis=1)

        # add 'media' column
        df['media'] = df['attachments.media'].apply(get_image_url)

        # optionally render clickable media images
        if incl_html:
            df['media'] = df['media'].apply(get_image_html)

        # add 'type' column (i.e., tweet type)
        df['type'] = df.apply(find_tweet_type, axis=1)

    # add 'tweet_count' column
    df['tweet_count'] = 1

    # rename some of the columns
    df = df.rename(
        columns={
            'author.name': 'name',
            'author.username': 'username',
            'author.verified': 'verified',
            'public_metrics.impression_count': 'impressions',  # added by Naomi
            'public_metrics.like_count': 'likes',
            'public_metrics.quote_count': 'quotes',
            'public_metrics.reply_count': 'replies',
            'public_metrics.retweet_count': 'retweets',
            'author.description': 'user_bio'
        }
    )

    #

    return df


# in place of no image (see `get_image_url()` & `get_image_html()`)
NO_IMAGE = 'No image URL'


def make_tweet_url(tweets):
    '''Make the tweet link clickable.'''
    # get username
    username = tweets[0]

    # get tweet IDs
    tweet_id = tweets[1]

    # make tweet URL
    return f"https://twitter.com/{username}/status/{tweet_id}"


def get_image_url(media):
    '''Extract the image URL.'''
    if not pd.isna(media):

        # convert `media` to a Python dictionary
        media = literal_eval(media)[0]

        # extract the media url if it exists
        return media.get('url', NO_IMAGE)

    return NO_IMAGE


def get_image_html(url):
    '''Make the image URL clickable.'''
    # check to see if the media category has an image URL
    if url != NO_IMAGE:

        # format the image url as an HTML image
        return f"<a href='{url}'>'<img src='{url}' width='500px'></a>"

    return url


def find_tweet_type(tweet):
    '''Find the tweet type.'''
    if not pd.isna(tweet['referenced_tweets.retweeted.id']):
        return "retweet"

    if not pd.isna(tweet['referenced_tweets.quoted.id']):

        if not pd.isna(tweet['referenced_tweets.replied_to.id']):
            return "quote/reply"

        return "quote"

    if not pd.isna(tweet['referenced_tweets.replied_to.id']):
        return "reply"

    return "original"


def create_dtypes():
    '''Create a column-to-dtype dictionary.'''
    # define the data types ('dtypes') used by Pandas;
    # see https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
    object_dtype = 'object'
    str_dtype = 'string'  # better than using 'object' when applicable
    int_dtype = 'Int64'
    float_dtype = 'Float64'

    # for simplicity, we'll cast enum types as strings (rather than 'category')
    enum_dtype = 'string'

    # we'll cast the datetime columns as objects for now, then convert them
    # into datetimes in pd/dd.read_csv()
    dt_dtype = 'object'

    # the 'boolean' (vs. 'bool') dtype is experimental, as it
    # adds supports for NA values:
    # https://pandas.pydata.org/docs/user_guide/boolean.html
    bool_dtype = 'boolean'

    # create the column-to-dtype dictionary; unless otherwise specified in an
    # inline comment, the following column names come directly from the Twitter
    # API (I think); here is nifty page on the Twitter API data types:
    # https://developer.twitter.com/en/docs/twitter-api/tweets/lookup/api-reference/get-tweets
    dtypes = {
        # objects
        'attachments.media': object_dtype,
        'attachments.media_keys': object_dtype,
        'attachments.poll.options': object_dtype,  # null
        'attachments.poll_ids': object_dtype,  # null
        'author.entities.description.cashtags': object_dtype,  # null
        'author.entities.description.hashtags': object_dtype,
        'author.entities.description.mentions': object_dtype,
        'author.entities.description.urls': object_dtype,
        'author.entities.url.urls': object_dtype,
        'author.withheld.country_codes': object_dtype,
        'context_annotations': object_dtype,
        'edit_history_tweet_ids': object_dtype,
        'entities.annotations': object_dtype,
        'entities.cashtags': object_dtype,
        'entities.hashtags': object_dtype,
        'entities.mentions': object_dtype,
        'entities.urls': object_dtype,
        'geo.coordinates.coordinates': object_dtype,
        'geo.geo.bbox': object_dtype,
        'matching_rules': object_dtype,  # null
        'withheld.country_codes': object_dtype,
        'Unnamed: 78': object_dtype,  # null -what is this unnamed column?

        # strings (n.b. casting id columns as strings resolves the scientific
        # notation issue)
        'attachments.poll.id': str_dtype,
        'attachments.poll.voting_status': str_dtype,
        'author.description': str_dtype,
        'author.id': str_dtype,
        'author.location': str_dtype,
        'author.name': str_dtype,
        'author.pinned_tweet_id': str_dtype,
        'author.profile_image_url': str_dtype,
        'author.url': str_dtype,
        'author.username': str_dtype,
        'author_id': str_dtype,
        'conversation_id': str_dtype,
        'geo.coordinates.type': str_dtype,
        'geo.country': str_dtype,
        'geo.country_code': str_dtype,
        'geo.full_name': str_dtype,
        'geo.geo.type': str_dtype,
        'geo.id': str_dtype,
        'geo.name': str_dtype,
        'geo.place_id': str_dtype,
        'geo.place_type': str_dtype,
        'id': str_dtype,
        'in_reply_to_user_id': str_dtype,
        'lang': str_dtype,
        'quoted_user_id': str_dtype,
        'referenced_tweets.quoted.id': str_dtype,
        'referenced_tweets.replied_to.id': str_dtype,
        'referenced_tweets.retweeted.id': str_dtype,
        'reply_settings': str_dtype,
        'retweeted_user_id': str_dtype,
        'source': str_dtype,
        'text': str_dtype,
        '__twarc.url': str_dtype,
        '__twarc.version': str_dtype,
        'media': str_dtype,         # added in format_df()
        'tweet_url': str_dtype,     # added in format_df()
        'username': str_dtype,      # name change from format_df()
        'user_bio': str_dtype,      # name change from format_df()

        # floats
        'attachments.poll.duration_minutes': float_dtype,

        # integers
        'author.public_metrics.followers_count': int_dtype,
        'author.public_metrics.following_count': int_dtype,
        'author.public_metrics.listed_count': int_dtype,
        'author.public_metrics.tweet_count': int_dtype,
        'edit_controls.edits_remaining': int_dtype,
        'public_metrics.impression_count': int_dtype,
        'public_metrics.like_count': int_dtype,
        'public_metrics.quote_count': int_dtype,
        'public_metrics.reply_count': int_dtype,
        'public_metrics.retweet_count': int_dtype,
        'tweet_count': int_dtype,   # added in format_df()
        'likes': int_dtype,         # name change from format_df()
        'quotes': int_dtype,        # name change from format_df()
        'replies': int_dtype,       # name change from format_df()
        'retweets': int_dtype,      # name change from format_df()

        # Booleans
        'author.protected': bool_dtype,
        'author.verified': bool_dtype,
        'author.withheld.copyright': bool_dtype,
        'edit_controls.is_edit_eligible': bool_dtype,
        'possibly_sensitive': bool_dtype,
        'withheld.copyright': bool_dtype,  # has NA values
        'verified': bool_dtype,     # name change from format_df()

        # enums
        'author.withheld.scope': enum_dtype,
        'withheld.scope': enum_dtype,
        'type': enum_dtype,         # added in format_df()

        # datetimes
        'attachments.poll.end_datetime': dt_dtype,
        'author.created_at': dt_dtype,
        'created_at': dt_dtype,
        'edit_controls.editable_until': dt_dtype,
        '__twarc.retrieved_at': dt_dtype,
        'date': dt_dtype,           # added in TWITTER_DT.set_date()
    }

    return dtypes


# data types ('dtypes') to pass to pd/dd.read_csv()
TWITTER_DTYPES = create_dtypes()


# helper object for handling Twitter datetime columns in pd/dd.read_csv() and
# pd/dd.to_csv()
class TWITTER_DT:

    # the strftime representation of Twitter timestamps (n.b. we'll want to use
    # this format for both reading and writing the data);
    # see also https://stackoverflow.com/questions/71100099/how-to-parse-twitter-api-v2-timestamps-to-python-datetime # noqa
    parse = '%Y-%m-%dT%H:%M:%S.%fZ'

    # the datetime columns from the Twitter API
    _columns = [
        'attachments.poll.end_datetime',
        'author.created_at',
        'created_at',
        'edit_controls.editable_until',
        '__twarc.retrieved_at',
    ]

    @property  # called as TWITTER_DT.columns
    def columns(self):
        '''Return the list of columns as a shallow copy.'''
        return self._columns.copy()

    @classmethod
    def add_date(cls, df):
        '''Add 'date' column to `df`.'''
        # convert the time portion of each 'date' to midnight;
        # see https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.normalize.html # noqa
        df['date'] = df['created_at'].dt.normalize()

        # make each 'date' timezone-unaware;
        # see https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.tz_localize.html # noqa
        df['date'] = df['date'].dt.tz_localize(None)

        return df

    @classmethod
    def incl_date(cls):
        # add the 'date' column
        cls._columns.append('date')

    @classmethod
    def remove_date(cls):
        # remove 'date' column
        try:
            cls._columns.remove('date')
        except ValueError:
            pass


TWITTER_DT = TWITTER_DT()


# helper object for loading datasets into Dask
class Data:

    # instantiate the Data() object with the path to thedirectory containing
    # the resampled data
    def __init__(self, resampled_dir='./data/baldwin/resampled/'):
        # path to the resampled data directory
        self.resampled_dir = resampled_dir

        # path to the S3 bucket containing the full Baldwin Parquet dataset;
        # TODO: define elsewhere / make Data() extensible
        self.full_parquet_dataset_dir = \
            's3://melwalshtweets/full_baldwin_tweets_2006-2023/'

        # configure S3-necessary things...
        self.aws_storage_options = {
            'key': config.AWS_ACCESS_KEY_ID,
            'secret': config.AWS_SECRET_ACCESS_KEY,
        }

    # full Twitter CSV dataframes ---------------------------------------------

    def read_twitter_csv_df(self, csv_path):
        '''Read a Twitter CSV dataset into Dask by passing in the path/URL.'''
        df = dd.read_csv(
            csv_path,
            assume_missing=True,
            low_memory=False,
            dtype=TWITTER_DTYPES,
            parse_dates=TWITTER_DT.columns,  # datetime columns
            date_format=TWITTER_DT.parse,    # datetime format
            )

        return df

    # Parquet dataframes ------------------------------------------------------

    def read_parquet_df(self, data_dir, storage_options=None):
        '''Load a Parquet dataset into Dask.'''
        df = dd.read_parquet(
            data_dir,
            engine='pyarrow',
            calculate_divisions=True,  # necessary for resampling
            assume_missing=True,
            storage_options=storage_options,
            )

        return df

    def get_full_df(self):
        '''Load the full Parquet dataset.'''
        dd = self.read_parquet_df(
            self.full_parquet_dataset_dir,
            storage_options=self.aws_storage_options,
            )

        return dd

    # resampled dataframes ----------------------------------------------------
    # see here for details on DateOffsets/rules:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

    def read_resampled_df(self, fn, index_col='date'):
        '''Read in resampled CSV dataframe.'''
        # compose filepath
        fn = op.join(self.resampled_dir, fn)

        # read in dataframe
        df = pd.read_csv(fn, index_col='date', parse_dates=True)

        return df

    def get_resampled_day_count_df(self, fn='day_tweet-count.csv'):
        '''Load tweet counts, resampled by Day.'''
        return self.read_resampled_df(fn)

    def get_resampled_day_type_count_df(self, fn='day_type_tweet-count.csv'):
        '''Load tweet counts, grouped by 'type' and resampled by Day.'''
        return self.read_resampled_df(fn)

    def get_resampled_day_retweet_count_df(self, fn='day_retweet-count.csv'):
        '''Load retweet counts, grouped by tweet and resampled by Day.'''
        return self.read_resampled_df(fn, index_col='retweet_date')
