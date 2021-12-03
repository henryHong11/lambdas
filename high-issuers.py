from __future__ import print_function
from io import StringIO
import boto3
from json import loads
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.svm import SVR
import json
import pandas as pd
import numpy as np
from boto3.dynamodb.conditions import Key, Attr
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import dateutil.parser as parser
from datetime import datetime
# from dynamodb_json import json_util as json
import pickle
from decimal import Decimal
from dateutil.parser import isoparse
from datetime import datetime as dt
import warnings
import time
from price_calc import run_yield_to_price
from botocore.config import Config
from config import min_amount, min_tenor, env_var
import requests
from utilities import download_sec, batch_get_table
import logging
import os

####### Logging Se
logger = logging.getLogger()
if env_var == "UAT":
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

config = Config(
    retries={
        'total_max_attempts': 10,
        'mode': 'standard'
    }
)
warnings.filterwarnings('ignore')
# from modeling.ml_function import benchmark_model
# print('Loading functions')
main_bucket = "nearest-neighbours-uat"
ddb = boto3.resource('dynamodb', config=config)


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return str(o)
        return super(DecimalEncoder, self).default(o)


"""
To DO
1- add refrence data to general
2- refrence will be the timestamp from latest quotes so remove the time stamp from
general liquidity
"""

col_scores_dict = {"score_vol": "price_volatility",
                   "score_bs_sp": "bid_ask_avg",
                   "score_liq_ratio": "liquidity_ratio"
                   }

s3_cache = {}


def download_ref(currency, bucket, high_issuer=True, ask=False):
    """
    This function will load the
    latest updated high issuers
    *** the bucket and data must be modified so either independent of datetime
    or the function must me mdodified in a way to capture the latest relevant file
    """
    global s3_cache
    s3_cache_key = currency + '-' + bucket
    if s3_cache_key in s3_cache and s3_cache[s3_cache_key]['last_updated'] > time.time() - 300:
        # print('Using cached S3 data.')
        df_ask = s3_cache[s3_cache_key]['ask']
        df_bid = s3_cache[s3_cache_key]['bid']
    else:
        # print('No S3 data found or expired. Fetching...')
        client = boto3.client('s3')
        if high_issuer:
            object_key_ask = f'jobs/_cobi/cobi-high-issuers/high_issuer__{currency}_ask.csv'

            object_key_bid = f'jobs/_cobi/cobi-high-issuers/high_issuer__{currency}.csv'

            csv_obj_ask = client.get_object(Bucket=bucket, Key=object_key_ask)
            csv_obj_bid = client.get_object(Bucket=bucket, Key=object_key_bid)

            body_ask = csv_obj_ask['Body']
            body_bid = csv_obj_bid['Body']
            csv_string_ask = body_ask.read().decode('utf-8')
            csv_string_bid = body_bid.read().decode('utf-8')
        df_ask = pd.read_csv(StringIO(csv_string_ask))
        df_bid = pd.read_csv(StringIO(csv_string_bid))

        s3_cache[s3_cache_key] = {
            'ask': df_ask,
            'bid': df_bid,
            'last_updated': time.time()
        }

    return list(
        set(df_ask['issuer_id'].astype('str').unique().tolist() + df_bid['issuer_id'].astype('str').unique().tolist()))


def valid_high_issuers(event, target_issuers=[], invoked=False):
    issuer_toupdate = set()
    # if not all_issuers:
    if invoked:
        for iss_id in event['high_issuers']:
            issuer_toupdate.add(iss_id)
        return issuer_toupdate, event['model_id']
    else:
        for data in event["Records"]:
            if data['eventName'] == "INSERT" or data['eventName'] == "MODIFY":
                iss_id = data['dynamodb']["NewImage"]["issuer_id"]['N']
                if iss_id in target_issuers:
                    issuer_toupdate.add(iss_id)
                    # print(f"******{iss_id} must be updated")

                else:
                    continue
                    # print(data['dynamodb']["NewImage"]["issuer_id"]['N'])
                    logger.info(f"******{iss_id} is Not a valid high issuer")
        return issuer_toupdate, None
    # return set(target_issuers)

    # return set(target_issuers)


# def get_table(table=None):
#     final_table = pd.DataFrame()
#     try:
#         js_tab = ddb.Table(table)
#         temp_tab = js_tab.scan() # it might need modification as size might get big in the future
#         final_table = pd.concat([final_table, pd.DataFrame(temp_tab['Items'])])
#         while 'LastEvaluatedKey' in temp_tab:
#             temp_tab = js_tab.scan(ExclusiveStartKey=temp_tab['LastEvaluatedKey'])
#             final_table = pd.concat([final_table, pd.DataFrame(temp_tab['Items'])])

#         #if 'LastEvaluatedKey' in temp_tab:
#         #    raise Exception("too much data to scan")
#         #final_table = pd.concat([final_table, pd.DataFrame(temp_tab['Items'])])

#     except Exception:
#         return "For some reason couldnt scan general_liquidity table"
#     return final_table

from boto3.dynamodb.conditions import Attr


def get_table(table=None, currency='EUR'):
    final_table = pd.DataFrame()
    try:
        js_tab = ddb.Table(table)
        temp_tab = js_tab.scan(FilterExpression=Attr('currency').eq(
            currency))  # it might need modification as size might get big in the future
        final_table = pd.concat([final_table, pd.DataFrame(temp_tab['Items'])])
        while 'LastEvaluatedKey' in temp_tab:
            temp_tab = js_tab.scan(ExclusiveStartKey=temp_tab['LastEvaluatedKey'],
                                   FilterExpression=Attr('currency').eq(currency))
            final_table = pd.concat([final_table, pd.DataFrame(temp_tab['Items'])])
        # if 'LastEvaluatedKey' in temp_tab:
        #    raise Exception("too much data to scan")
        # final_table = pd.concat([final_table, pd.DataFrame(temp_tab['Items'])])
    except Exception as e:
        logger.error(e)
        logger.info(type(e).__name__)
        logger.info("For some reason couldnt scan general_liquidity table")
    return final_table


def liquidty_check(df):
    """
    This function will run simple percentile based
    liquiidty check and returns a list of outliers
    """
    max_avr = np.percentile(df.bid_ask_avg, q=88)
    max_high_low = np.percentile(df.high_low_diff, q=89)
    # max_var = np.percentile(main_data.var_bid_ask, q=88)
    removed = df[(df.high_low_diff > max_high_low) |
                 (df.bid_ask_avg > max_avr)]['identifier'].tolist()
    return removed


def norm_func(df, col, qt_cnt=False):
    """
    This function does bot normalization
    and scoring
    """
    pw = PowerTransformer()
    MX = MinMaxScaler(feature_range=(0, 1))
    scaled_pw = pw.fit_transform(df[[col]])
    scaled_min = MX.fit_transform(scaled_pw)
    if not qt_cnt:
        scaled_min = 1 - scaled_min
    return scaled_min


def load_latest_model(Ispread=False, Gspread=False, currency="EUR", bid=True):
    """
    this function will load the latest model for ispread and/or Gspread
    if EUR will have to load both g and i models otherwise just g-spread
    """
    model_table = ddb.Table('test_benchmarks')
    if Ispread:  # must change
        if bid:
            item = model_table.get_item(Key={'id': 'B-i-EUR'})['Item']
        else:
            item = model_table.get_item(Key={'id': 'A-i-EUR'})['Item']

        mod_binary = item['model'].value
        latest_m_t = item['source_timestamp']
        model = pickle.loads(mod_binary)
        return model, latest_m_t
    elif Gspread:  # MUST CHANGE
        if bid:
            item = model_table.get_item(Key={'id': f"B-g-{currency}"})['Item']
        else:
            item = model_table.get_item(Key={'id': f"A-g-{currency}"})['Item']
        mod_binary = item['model'].value
        latest_m_t = item['source_timestamp']
        model = pickle.loads(mod_binary)
        return model, latest_m_t


def backout_spread(model=None,
                   df=None, col=None, main_col="bid_yield"):
    """
    use the benchmark model and raw bid yield to calculate
    raw spreas (I or G)
    """
    benchmarks = model.predict(df[[col]])
    return (df[main_col].astype('float') - benchmarks), benchmarks


def training(df=None, x="tenor", y="bid_yield"):
    """
    This function trains a SVR model on latest quotes
    predict on same quotes
    takes average
    calculate the error (avg -raw input)
    """
    # model = SVR(C=5e2, gamma=4.6e-3, epsilon=0.7, kernel='rbf',).fit(df[[x]],df[[y]])
    model = SVR(C=5e2, gamma=4.6e-3, epsilon=0.1, kernel='rbf', ).fit(df[[x]], df[[y]])
    avg = (model.predict(df[[x]]) * 0.3 + df[y] * 0.7)  # / 2
    err = np.abs(avg - df[y])
    return avg, err, model


def decimal_default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError


# def update_results(identifier, final_df
#                   ):
#     result_table = ddb.Table('results')
#     cols_drop = ["maturity_date", "call_date", "call_price", "callable", "coupon_rate", "I_err", "G_err"]
#     for c_drp in cols_drop:
#         if c_drp in final_df.columns:
#             final_df.drop(c_drp, axis=1, inplace=True)
#     # test_records = final_df[final_df.identifier == identifier].to_dict(orient='Records')
#     final_df = final_df.fillna('None')
#     final_df = final_df[final_df.identifier == identifier]
#     final_df['ric'] = final_df['identifier']
#     final_df['identifier'] = final_df['isin']

#     test_records = final_df.to_dict(orient='Records')
#     print("*" * 50, "START LOOPING")
#     for record in test_records:
#         update_expr = "SET "
#         expr_attribute = {}
#         for key, value in record.items():
#             if key != 'identifier':
#                 update_expr += f"{key} = :{key},"
#                 if isinstance(value, float):
#                     value = Decimal(str(value))
#                 #   if str(value)=='nan' or str(value)=='NaN':
#                 #       value = 'None'
#                 expr_attribute[f":{key}"] = value

#         update_expr = update_expr[:-1]
#         expr_attribute2 = {}
#         for key2 in expr_attribute.keys():
#             expr_attribute2[key2[1:]] = expr_attribute[key2]
#         record_load = {'recordKey': record['identifier']}
#         record_load['payload'] = expr_attribute2
#         pload = {'apiKey': 'WcsIDY2hLh7IBJ4HiE1JVSLjHaSnY3SnxGciUoHP', 'records': [record_load]}
#         # print(pload)
#         print("*" * 50, "START REQUEST")
#         requests.post('http://wsfwd.eba-8aiqey9p.us-east-1.elasticbeanstalk.com/',
#                       data=json.dumps(pload, default=decimal_default), headers={'Content-type': 'application/json'})
#         print("*" * 50, "FINISH REQUEST")
#         try:
#             update_results = result_table.update_item(Key={'identifier': record['identifier']},
#                                                       UpdateExpression=update_expr,
#                                                       ExpressionAttributeValues=expr_attribute)
#         except ddb.meta.client.exceptions.ProvisionedThroughputExceededException as e:
#             print(e)
#             curwritecap = result_table.provisioned_throughput['WriteCapacityUnits']
#             curreadcap = result_table.provisioned_throughput['ReadCapacityUnits']
#             update_table = result_table.update(ProvisionedThroughput={'ReadCapacityUnits': curreadcap,
#                                                                       'WriteCapacityUnits': round(curwritecap * 1.125)})
#             print("waiting")
#             client = ddb.meta.client
#             status = client.describe_table(TableName=result_table.name)['Table']['TableStatus']
#             while status != 'ACTIVE':
#                 status = client.describe_table(TableName=result_table.name)['Table']['TableStatus']
#                 time.sleep(0.5)

#             update_results = result_table.update_item(Key={'identifier': record['identifier']},
#                                                       UpdateExpression=update_expr,
#                                                       ExpressionAttributeValues=expr_attribute)
#         except Exception as e:
#             print(e)
#             print("uncaught exception")

#         # print(5*"update results **")


def update_batch(final_df
                 ):
    # t0=dt.now()
    result_table = ddb.Table('results')
    cols_drop = ["maturity_date", "call_date", "call_price", "callable", "coupon_rate", "I_err", "G_err"]
    for c_drp in cols_drop:
        if c_drp in final_df.columns:
            final_df.drop(c_drp, axis=1, inplace=True)
    final_df = final_df.fillna('None')
    final_df['ric'] = final_df['identifier']
    final_df['identifier'] = final_df['isin']

    test_records = final_df.to_dict(orient='Records')
    # update_expr = "SET "
    # expr_attribute = {}
    # t4=dt.now()
    records = []
    db_updates = []
    for record in test_records:
        update_expr = "SET "
        expr_attribute = {}
        for key, value in record.items():
            if (key != 'identifier'):
                update_expr += f"{key} = :{key},"
                # if isinstance(value, decimal) :
                #     value = float(str(value))

                if str(value) == 'nan' or str(value) == 'NaN':
                    value = 'None'
                expr_attribute[f":{key}"] = value

        update_expr = update_expr[:-1]
        # print(record['identifier'])
        # print(update_expr)
        expr_attribute2 = {}
        # print("Newwwww",record['identifier'])
        for key2 in expr_attribute.keys():
            expr_attribute2[key2[1:]] = expr_attribute[key2]
        record_load = {'recordKey': record['identifier']}
        record_load['payload'] = expr_attribute2
        records.append(record_load)
        # print(records)
        db_updates.append({'id': record['identifier'], 'update_expr': update_expr, 'expr_attribute': expr_attribute})
    pload = {'apiKey': 'WcsIDY2hLh7IBJ4HiE1JVSLjHaSnY3SnxGciUoHP', 'records': records}
    json_pload = json.dumps(pload, cls=DecimalEncoder)
    x = requests.post('http://wsfwd.eba-8aiqey9p.us-east-1.elasticbeanstalk.com/', data=json_pload,
                      headers={'Content-type': 'application/json'})
    # print(x.status_code, "STATUS")
    # print("Start Batch eriting for high issuers")
    # print(test_records)
    with result_table.batch_writer(overwrite_by_pkeys=["identifier", ]) as batch:
        for record in test_records:

            expr_attribute = {}
            for key, value in record.items():
                if isinstance(value, float):
                    value = Decimal(str(value))
                if str(value) == 'nan' or str(value) == 'NaN':
                    value = 'None'
                expr_attribute[f"{key}"] = value
            batch.put_item(
                Item=expr_attribute)
    logger.info("It seems batch is working for High issuers")


def update_high_issuers_models(issuer_id, model, Ispread=True, currency="EUR", ts=None, bid=True):
    """
    put the latest high issuer model in high_issuer_models ddb table
    key will be issuer_id
    """
    model_table = ddb.Table('high_issuer_models_test')
    if Ispread:
        if bid:
            model_id = f"B-{issuer_id}-{currency}-I"

        else:
            model_id = f"A-{issuer_id}-{currency}-I"


    else:
        if bid:
            model_id = f"B-{issuer_id}-{currency}-G"
        else:
            model_id = f"A-{issuer_id}-{currency}-G"
    # print(ts)
    if isinstance(ts, str):
        # print("ts is string")
        source_ts = parser.parse(ts).isoformat()
    elif isinstance(ts, dt) or ts is pd.Timestamp:
        # print("ts is date")
        source_ts = ts.isoformat()
    else:
        raise Exception(" cant determine type of ts")
    try:
        query_db = model_table.get_item(Key={'model_id': model_id})
    except ddb.meta.client.exceptions.ProvisionedThroughputExceededException as e:
        logger.error(e)
        curwritecap = model_table.provisioned_throughput['WriteCapacityUnits']
        curreadcap = model_table.provisioned_throughput['ReadCapacityUnits']
        update_table = model_table.update(ProvisionedThroughput={'ReadCapacityUnits': round(curreadcap * 1.125),
                                                                 'WriteCapacityUnits': (curwritecap)})
        # print("waiting")
        client = ddb.meta.client
        status = client.describe_table(TableName=model_table.name)['Table']['TableStatus']
        while status != 'ACTIVE':
            status = client.describe_table(TableName=model_table.name)['Table']['TableStatus']
            time.sleep(0.5)
        query_db = model_table.get_item(Key={'model_id': model_id})
    except Exception as e:
        logger.error(e)
        logger.info("un caught exception")

    ts = datetime.now()
    if 'Item' in query_db:
        latest_ts = query_db['Item']['model_timestamp']
        if isoparse(latest_ts) < ts:
            try:
                # print("updating mode")
                final_model = (model_table.update_item(Key={'model_id': model_id},
                                                       UpdateExpression="SET model = :model,model_timestamp = :t,source_timestamp = :d",
                                                       ExpressionAttributeValues={':model': pickle.dumps(model),
                                                                                  ':t': ts.isoformat(),
                                                                                  ':d': source_ts
                                                                                  }))
                return model_id
            except ddb.meta.client.exceptions.ProvisionedThroughputExceededException as e:
                logger.error(e)
                curwritecap = model_table.provisioned_throughput['WriteCapacityUnits']
                curreadcap = model_table.provisioned_throughput['ReadCapacityUnits']
                update_table = model_table.update(ProvisionedThroughput={'ReadCapacityUnits': curreadcap,
                                                                         'WriteCapacityUnits': round(
                                                                             curwritecap * 1.175)})

                # print("waiting")
                client = ddb.meta.client
                status = client.describe_table(TableName=model_table.name)['Table']['TableStatus']
                while status != 'ACTIVE':
                    status = client.describe_table(TableName=model_table.name)['Table']['TableStatus']
                    time.sleep(0.5)

                final_model = (model_table.update_item(Key={'model_id': model_id},
                                                       UpdateExpression="SET model = :model,model_timestamp = :t,source_timestamp = :d",
                                                       ExpressionAttributeValues={':model': pickle.dumps(model),
                                                                                  ':t': ts.isoformat(),
                                                                                  ':d': source_ts
                                                                                  }))
                return model_id
            except Exception as e:
                logger.error(e)
                logger.info("un caught exception")
                return None
        else:
            # print(f"table timestamp is {latest_ts}")
            # print(f"current time is {ts}")
            # print("model not updated")
            return None
    else:
        try:
            logger.info("item doesnt exist, creating new row")
            final_model = (model_table.update_item(Key={'model_id': model_id},
                                                   UpdateExpression="SET model = :model,model_timestamp = :t,source_timestamp = :d",
                                                   ExpressionAttributeValues={':model': pickle.dumps(model),
                                                                              ':t': ts.isoformat(),
                                                                              ':d': source_ts
                                                                              }))
            return model_id
        except ddb.meta.client.exceptions.ProvisionedThroughputExceededException as e:
            logger.error(e)
            curwritecap = model_table.provisioned_throughput['WriteCapacityUnits']
            curreadcap = model_table.provisioned_throughput['ReadCapacityUnits']
            update_table = model_table.update(ProvisionedThroughput={'ReadCapacityUnits': curreadcap,
                                                                     'WriteCapacityUnits': round(curwritecap * 1.175)})
            # print("waiting")
            client = ddb.meta.client
            status = client.describe_table(TableName=model_table.name)['Table']['TableStatus']
            while status != 'ACTIVE':
                status = client.describe_table(TableName=model_table.name)['Table']['TableStatus']
                time.sleep(0.5)

            final_model = (model_table.update_item(Key={'model_id': model_id},
                                                   UpdateExpression="SET model = :model,model_timestamp = :t,source_timestamp = :d",
                                                   ExpressionAttributeValues={':model': pickle.dumps(model),
                                                                              ':t': ts.isoformat(),
                                                                              ':d': source_ts
                                                                              }))
            return model_id
        except Exception as e:
            logger.error(e)
            logger.info("un caught exception")
            return None

        # print(model)


def can_recommend(main_data, identifier, min_score, mid_score, max_score, col="conf_sc_num"):
    """
    somehow this function makes sure to update whatever
    identifier that falls into tier-2 region
    """
    main_data = main_data[main_data.identifier == identifier].copy()
    val = main_data.loc[:, col].values[0]
    # print(f"tier_2_score is {mid_score} tier_1_score {max_score} min_score {min_score} conf score is {val}")
    # print("mainnnnnn dataaaa",main_data)
    if (val > mid_score) & (val < max_score):
        # print(f"recommendation is {1} for {identifier}")
        return True, 1, main_data
    elif (val > min_score) & (val <= mid_score):
        # print(f"recommendation is {0} for {identifier}")
        return True, 2, main_data
    elif (val < min_score):  # here we make sure not touch tier-1
        # print(f"recommendation is {0} for {identifier}")
        return False, None, None
    else:
        return False, None, None


def update_table_per_item(main_data, identifier, ispread=True,
                          recommendation=None):  # this needs to be modified for batch
    row = main_data.copy()
    if ispread:
        if recommendation >= 1:
            if row['I_err'].values <= 1.1:
                row.loc[:, "recommendation"] = recommendation
            else:
                row.loc[:, "recommendation"] = 0
        else:
            row.loc[:, "recommendation"] = 0
        # print("Updating the results for Ispread")
        # print(row['trade_timestamp'])
        row = row.fillna('None')
        update_results(identifier, row)
    else:
        if recommendation >= 1:
            if row['G_err'].values <= 1.1:
                row.loc[:, "recommendation"] = recommendation
            else:
                row.loc[:, "recommendation"] = 0
        else:
            row.loc[:, "recommendation"] = 0
        # print("Updating the results for Ispread")
        # print(row['trade_timestamp'])
        row = row.fillna('None')
        update_results(identifier, row)


def update_table_per_item_batch(main_data, identifier, ispread=True,
                                recommendation=None):  # this needs to be modified for batch
    row = main_data.copy()
    if ispread:
        if recommendation >= 1:
            if row['I_err'].values <= 1.3:
                row.loc[:, "recommendation"] = recommendation
            else:
                row.loc[:, "recommendation"] = 0
        else:
            row.loc[:, "recommendation"] = 0
        # print("Updating the results for Ispread")
        # print(row['trade_timestamp'])
        row = row.fillna('None')
        # update_results(identifier, row)
        return row
    else:
        if recommendation >= 1:
            if row['G_err'].values <= 1.3:
                row.loc[:, "recommendation"] = recommendation
            else:
                row.loc[:, "recommendation"] = 0
        else:
            row.loc[:, "recommendation"] = 0
        # print("Updating the results for Ispread")
        # print(row['trade_timestamp'])
        row = row.fillna('None')
        # update_results(identifier, row)
        return row


def adjust_conf(data):
    time_now = dt.now()
    data['delta'] = (time_now - data['trade_timestamp']).dt.days + 1
    data['conf_sc_num'] = data['conf_sc_num'] / data['delta']
    data.drop("delta", axis=1, inplace=True)
    return data


def common_trigger(event, context, all_issuers=False, trigger_name="", invoked=False, currency=""):
    t0 = dt.now()
    # print(f"Time now is {t0}")
    issuer_liq_table = get_table(table="issuer_liquidity", currency=currency)
    # issuer_liq_table['trade_timestamp'] = pd.to_datetime(issuer_liq_table['trade_timestamp'])
    # print("latest time issuer_liquidity",issuer_liq_table[["identifier","trade_timestamp"]].sort_values(by="trade_timestamp", ascending=False).head(1))
    latest_quotes = get_table(table="market_data_quotes", currency=currency)
    # print("test Batch get table")
    latest_quotes = latest_quotes[latest_quotes.current_amount_outstanding >= min_amount[currency]]
    # print(f"Length get table old is {len(latest_quotes)}")

    ####################### TEST BATCH

    # isn_list_batch = download_sec(currency=currency)
    # print(f"rics {len(isn_list_batch)}")
    # new_lt_quotes = batch_get_table(table="market_data_quotes",isin_list=isn_list_batch)
    # print(f"Length get table batch is {len(new_lt_quotes)}")

    ###################### END TEST BATCH
    latest_quotes = latest_quotes[latest_quotes.tenor >= min_tenor]
    latest_quotes['M_B_Price'] = latest_quotes['bid_price']
    latest_quotes['M_A_Price'] = latest_quotes['ask_price']
    # latest_quotes['trade_timestamp'] = pd.to_datetime(latest_quotes['trade_timestamp'])
    # print("latest time markets",latest_quotes[["identifier","trade_timestamp"]].sort_values(by="trade_timestamp", ascending=False).head(1))
    gen_liq_table = get_table(table="general_liquidity", currency=currency)
    gen_liq_table['trade_timestamp'] = pd.to_datetime(gen_liq_table['trade_timestamp'])
    t_load = dt.now()
    # print("Time spent to load all tables from ddb is : ", (t_load - t0).total_seconds())
    # gen_liq_table['trade_timestamp'] = pd.to_datetime(gen_liq_table['trade_timestamp'])
    # print("latest time general_liquidity",gen_liq_table[["identifier","trade_timestamp"]].sort_values(by="trade_timestamp", ascending=False).head(1))
    ha_list = download_ref(currency=currency, bucket=main_bucket, high_issuer=True)
    # print(ha_list)
    latest_gl_time = gen_liq_table['trade_timestamp'].max()
    latest_isl_time = issuer_liq_table['trade_timestamp'].max()

    if all_issuers:
        issuer_toupdate = set(ha_list)

    else:
        issuer_toupdate, model_bench = valid_high_issuers(event, ha_list, invoked=invoked)
        # print("Model Id is ", model_bench)

    # print(currency, issuer_toupdate)
    issuer_liq_table.issuer_id = issuer_liq_table.issuer_id.astype(str)

    # *** scoring
    gen_liq_table.issuer_id = gen_liq_table.issuer_id.astype(str)
    latest_quotes.issuer_id = latest_quotes.issuer_id.astype(str)
    gen_liq_table['price_volatility'] = (gen_liq_table['price_volatility'] + gen_liq_table['ask_volatility']) / 2
    for col_name, val_name in col_scores_dict.items():
        gen_liq_table[col_name] = norm_func(df=gen_liq_table.copy(), col=val_name, qt_cnt=False)
    gen_liq_table["score_qt_cnt"] = norm_func(df=gen_liq_table.copy(), col="qt_count", qt_cnt=True)
    gen_liq_table['conf_sc_num'] = (gen_liq_table["score_bs_sp"] + gen_liq_table["score_vol"]) / 2
    gen_liq_table['conf_sc_num'] = gen_liq_table['conf_sc_num'].round(2)

    # print(gen_liq_table[['conf_sc_num']].head(3), "conf score before adjustment")
    gen_liq_table = adjust_conf(
        gen_liq_table)  # here we added the time limit to adjust the confidence score based on the quote age
    # print(gen_liq_table[['conf_sc_num']].head(3), "conf score after adjustment")

    gen_liq_table['recommendation'], gen_liq_table['tier'] = 0, 2

    max_score = gen_liq_table['conf_sc_num'].quantile(0.75)
    mid_score = gen_liq_table['conf_sc_num'].quantile(0.55)
    # min_score = gen_liq_table['conf_sc_num'].quantile(0.40)
    min_score = gen_liq_table['conf_sc_num'].quantile(0.472)

    # print(f"**** max_score is {max_score} ** mid_score is : {mid_score} and min score is : {min_score}")
    # **** print(liq_table.columns,"columns")
    # print(issuer_toupdate)
    ha_liq_df = issuer_liq_table[issuer_liq_table.issuer_id.isin(issuer_toupdate)]  # high-issuer liquidity table
    ha_liq_df['bid_ask_avg'] = ha_liq_df['bid_ask_avg'].astype('float')
    ha_liq_df['high_low_diff'] = ha_liq_df['high_low_diff'].astype('float')
    # ha_liq_df.issuer_id = ha_liq_df.issuer_id.astype(str)
    latest_quotes.drop(['bid_price', 'ask_price'], axis=1, inplace=True)
    latest_ha_quotes = latest_quotes[
        latest_quotes.issuer_id.isin(issuer_toupdate)]  # latest quotes for valid updated high issuers

    # print("* * *length",len(latest_ha_quotes) )
    t1 = dt.now()
    time_spent1 = (t1 - t0).total_seconds()
    # print(f" total {time_spent1} seconds spent to prepare all tables and perform liquidity scores")
    # print(f" df size for latest quotes is {latest_quotes.shape}")
    # print(f" df size for high issuer df is {latest_ha_quotes.shape}")
    # print(len(latest_ha_quotes))
    ha_dict_invk = {}
    if len(latest_ha_quotes) > 5:  # this condition must be modified in second iteration

        tl1 = dt.now()
        # print(f"number of issuers to check is {len(issuer_toupdate)}")
        time_tot = 0.0
        end_df_ha = pd.DataFrame()
        for ha_id in issuer_toupdate:  # maybe paralle processing
            ti = dt.now()
            try:
                cond1 = (ha_liq_df.issuer_id == ha_id)
                cond2 = (latest_ha_quotes.issuer_id == ha_id)

                # print(f"*Here {ha_id} is getting updated")
                if len(ha_liq_df[cond1]) < 5:
                    logger.info("Not enough data for this high issuer")
                    continue
                outliers = liquidty_check(ha_liq_df[cond1])
                cond3 = (~latest_ha_quotes.identifier.isin(outliers))
                # print(f"*Here {outliers} are outliers")
                ha_df = latest_ha_quotes[cond2 & cond3].sort_values(by='tenor')
                if len(ha_df) < 5:
                    continue
                # ha_df.sort_values(by='tenor', inplace =True)
                # print(f"*Here {len(ha_df)} are number of bonds")

                #### must add ask and bid benchmark models
                tm0 = dt.now()
                if currency == "EUR":
                    model_is_B, latest_m_t_is = load_latest_model(Ispread=True, Gspread=False, currency=currency,
                                                                  bid=True)
                    model_is_A, _ = load_latest_model(Ispread=True, Gspread=False, currency=currency, bid=False)
                model_gs_B, latest_m_t_g = load_latest_model(Ispread=False, Gspread=True, currency=currency, bid=True)
                model_gs_A, _ = load_latest_model(Ispread=False, Gspread=True, currency=currency, bid=False)
                tm1 = dt.now()
                # print("Time spent to load EUR models from ddb is : ", (tm1 - tm0).total_seconds())
                if trigger_name == "issuer_liquidity":
                    sc_time = latest_isl_time
                elif trigger_name == 'general_liquidity':
                    sc_time = latest_gl_time

                elif trigger_name == "benchmark_models":
                    if currency == "EUR":
                        if latest_m_t_g > latest_m_t_is:  # if ispread is latest trigger
                            sc_time = latest_m_t_g
                        else:
                            sc_time = latest_m_t_is
                    else:  # if currency is not Eur so no timing for Ispread
                        sc_time = latest_m_t_g
                ha_df['source_timestamp'] = sc_time
                if currency == "EUR":  # do the Ispread if currency is EUR
                    # model_is,latest_m_t_is  = load_latest_model(Ispread = True, Gspread=False, currency=currency)
                    Ispreads, Ispread_benchmarks = backout_spread(model=model_is_B,
                                                                  df=ha_df.copy(), col="tenor", main_col="bid_yield")
                    ha_df['M_I_spread'], ha_df['ispread_benchmark'] = Ispreads, Ispread_benchmarks
                    ha_df["I_spread"], ha_df['I_err'], imodel = training(df=ha_df, x="tenor", y="M_I_spread")
                    ha_model_id_i_b = update_high_issuers_models(issuer_id=ha_id, model=imodel, Ispread=True,
                                                                 currency=currency, ts=sc_time)

                    ask_Ispreads, ask_Ispread_benchmarks = backout_spread(model=model_is_A,
                                                                          df=ha_df.copy(), col="tenor",
                                                                          main_col="ask_yield")
                    ha_df['M_ask_I_spread'], ha_df['ask_ispread_benchmark'] = ask_Ispreads, ask_Ispread_benchmarks
                    ha_df["ask_I_spread"], ha_df['ask_I_err'], ask_imodel = training(df=ha_df, x="tenor",
                                                                                     y="M_ask_I_spread")
                    ha_model_id_i_a = update_high_issuers_models(issuer_id=ha_id, model=ask_imodel, Ispread=True,
                                                                 currency=currency, ts=sc_time, bid=False)
                    if ha_model_id_i_b is not None:
                        ha_dict_invk[ha_model_id_i_b] = ha_id
                        ha_dict_invk[ha_model_id_i_a] = ha_id
                # do the Gspread for all currencies
                # model_gs,latest_m_t_g = load_latest_model(Ispread = False, Gspread=True, currency=currency)
                Gspreads, Gspread_benchmarks = backout_spread(model=model_gs_B,
                                                              df=ha_df.copy(), col="tenor", main_col="bid_yield")
                ha_df['M_G_spread'], ha_df['gspread_benchmark'] = Gspreads, Gspread_benchmarks
                ha_df["G_spread"], ha_df['G_err'], gmodel = training(df=ha_df, x="tenor", y="M_G_spread")
                ha_model_id_g_b = update_high_issuers_models(issuer_id=ha_id, model=gmodel, Ispread=False,
                                                             currency=currency, ts=sc_time)

                ask_Gspreads, ask_Gspread_benchmarks = backout_spread(model=model_gs_A,
                                                                      df=ha_df.copy(), col="tenor",
                                                                      main_col="ask_yield")
                ha_df['M_ask_G_spread'], ha_df['ask_gspread_benchmark'] = ask_Gspreads, ask_Gspread_benchmarks
                ha_df["ask_G_spread"], ha_df['ask_G_err'], ask_gmodel = training(df=ha_df, x="tenor",
                                                                                 y="M_ask_G_spread")
                ha_model_id_g_a = update_high_issuers_models(issuer_id=ha_id, model=ask_gmodel, Ispread=False,
                                                             currency=currency, ts=sc_time, bid=False)
                # print(
                #     f"currency is {currency} and BID ERROR for {ha_id} is {np.mean(ha_df['G_err'])} and ASK ERROR for {ha_id} is {np.mean(ha_df['ask_G_err'])}")
                if ha_model_id_g_b is not None:
                    ha_dict_invk[ha_model_id_g_b] = ha_id
                    ha_dict_invk[ha_model_id_g_a] = ha_id
                # print(ha_df['G_err'], ha_df['I_err'])
                merged_file = ha_df.merge(gen_liq_table, on=['identifier', 'isin',
                                                             'name']).drop(['tenor_x', 'sector_name_x',
                                                                            "trade_timestamp_y", "maturity_date_x",
                                                                            "currency_x"],
                                                                           axis=1)
                merged_file.rename(columns={'tenor_y': 'tenor',
                                            "sector_name_y": 'sector_name',
                                            "trade_timestamp_x": "trade_timestamp", "name": "issuer_name",
                                            "maturity_date_y": "maturity_date", "currency_y": "currency"}, inplace=True)
                # following analytic time should get improved, it may unncessary time
                ha_idfs = merged_file['identifier'].unique().tolist()
                # curr_time = time.localtime()
                # current_time = time.strftime("%H:%M:%S", curr_time)
                # merged_file['analytics_time'] = time.strftime("%H:%M:%S", time.localtime())  # current_time
                # merged_file['analytics_time'] = pd.to_datetime(merged_file['analytics_time'])
                merged_file['analytics_time'] = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
                merged_file['analytics_time'] = pd.to_datetime(merged_file['analytics_time'])
                merged_file['trigger_name'] = trigger_name
                if currency == "EUR":
                    # here we have to do bid_yield for different currencies, if EUR we just use Ispread+benchmark
                    merged_file['bid_yield'] = merged_file['ispread_benchmark'] + merged_file['I_spread']
                    merged_file['ask_yield'] = merged_file['ask_ispread_benchmark'] + merged_file['ask_I_spread']
                else:
                    merged_file['bid_yield'] = merged_file['gspread_benchmark'] + merged_file['G_spread']
                    merged_file['ask_yield'] = merged_file['ask_gspread_benchmark'] + merged_file['ask_G_spread']
                merged_file['conf_sc_num'] = np.round(merged_file['conf_sc_num'].astype('float'), 3)
                merged_file['score_liq_ratio'] = np.round(merged_file['score_liq_ratio'].astype('float'), 3)
                # merged_file['bid_price'] = np.round(merged_file['bid_price'].astype('float'),3)
                # merged_file['ask_price'] = np.round(merged_file['ask_price'].astype('float'),3)
                merged_file['bid_yield'] = np.round(merged_file['bid_yield'].astype('float'), 3)
                merged_file['ask_yield'] = np.round(merged_file['ask_yield'].astype('float'), 3)
                for col in merged_file.columns:

                    if merged_file.dtypes[col] == np.float64:
                        merged_file[col] = merged_file[col].astype(str)
                        merged_file[col] = merged_file[col].apply(lambda x: Decimal(x))
                    if merged_file.dtypes[col] == np.datetime64 or is_datetime(
                            merged_file[col]):  # or col=='analytics_time':
                        merged_file[col] = merged_file[col].apply(lambda x: x.isoformat())
                    if col == 'analytics_time':
                        merged_file[col] = merged_file[col].apply(lambda x: parser.parse(x).isoformat())
                cols_drop = [
                    "sector_name",
                    "issue_date", "issuer_id_x", "issuer_id", "issuer_id_y",
                    "score_vol", "score_bs_sp", "score_qt_cnt", "bid_ask_avg", "price_volatility",
                    "liquidity_ratio", "qt_count", "ask_ispread_benchmark", "ask_gspread_benchmark"]
                for c_drp in cols_drop:
                    if c_drp in merged_file.columns:
                        merged_file.drop(c_drp, axis=1, inplace=True)
                if currency == "EUR":  # this needs to be modifed for batch storage
                    for h_idf in ha_idfs:  # we have find a better way than this
                        can_rec, rec, main_df = can_recommend(main_data=merged_file, identifier=h_idf,
                                                              min_score=min_score, mid_score=mid_score,
                                                              max_score=max_score, col="conf_sc_num")
                        if can_rec:
                            # print(h_idf)
                            try:
                                main_df = run_yield_to_price(main_df, yield_column='bid_yield',
                                                             price_return_column='bid_price')

                                main_df = run_yield_to_price(main_df, yield_column='ask_yield',
                                                             price_return_column='ask_price')
                                # print(main_df.head())
                            except Exception as e:
                                logger.error(e)
                                continue
                            tu1 = dt.now()

                            # update_table_per_item(main_df, identifier=h_idf, ispread=True, recommendation=rec)
                            temp_row = update_table_per_item_batch(main_df, identifier=h_idf, ispread=True,
                                                                   recommendation=rec)
                            end_df_ha = pd.concat([end_df_ha, temp_row])

                            time_tot += (dt.now() - tu1).total_seconds()

                    # print("use ispread_error to update recommendation")
                else:  # this needs to be modified for batch
                    for h_idf in ha_idfs:  # we have find a better way than this
                        can_rec, rec, main_df = can_recommend(main_data=merged_file, identifier=h_idf,
                                                              min_score=min_score, mid_score=mid_score,
                                                              max_score=max_score, col="conf_sc_num")
                        if can_rec:
                            # print(h_idf)
                            try:
                                main_df = run_yield_to_price(main_df, yield_column='bid_yield',
                                                             price_return_column='bid_price')
                                main_df = run_yield_to_price(main_df, yield_column='ask_yield',
                                                             price_return_column='ask_price')
                                # print(main_df.head())
                            except Exception as e:
                                logger.error(f"Something wrong in price calculation : {e}")
                                continue

                            tu1 = dt.now()
                            for c_drp in cols_drop:
                                if c_drp in main_df.columns:
                                    main_df.drop(c_drp, axis=1, inplace=True)
                            # update_table_per_item(main_df, identifier=h_idf, ispread=False, recommendation=rec)
                            temp_row = update_table_per_item_batch(main_df, identifier=h_idf, ispread=False,
                                                                   recommendation=rec)
                            end_df_ha = pd.concat([end_df_ha, temp_row])
                            time_tot += (dt.now() - tu1).total_seconds()
                    # print("use G-spread error to update recommendation")



            except ValueError as e:
                logger.error(f"This Error happend when updating {ha_id} :: ", e)
            # print(f"total {(dt.now() - ti).total_seconds()} spent for {ha_id} with {len(ha_df)}")
        t_s1 = dt.now()
        if len(end_df_ha) >= 1:
            update_batch(end_df_ha)
        t_s2 = dt.now()
        # print(f"total {(t_s2 - t_s1).total_seconds()} spent for BATCH WRTING for {len(end_df_ha)} Bonds")

        tl2 = dt.now()
        # if time_tot:
        #     print(f"Total time spend on updating results is {time_tot}")
        # print(f"total {(tl2 - tl1).total_seconds()} spent for the for loop")
        # print(ha_dict_invk)
    return ha_dict_invk


# def inokeFunc(trigger_name="", ha_dict_invk={}, currency=""):
#     """
#     This function will invoke the low issuer function in parallel
#     It passes a dictionary of high issuers with their model_id as key and
#     issuer id as value
#     """
#     high_issuer_lst = list(ha_dict_invk.keys())
#     print(high_issuer_lst)
#     print(f"currency to invoke is {currency}")
#     if trigger_name == "benchmark_models":
#         client = boto3.client('lambda')
#         payload = {'high_issuers': ha_dict_invk, 'currency': [currency]}
#         print(payload)
#         response = client.invoke(
#             FunctionName='arn:aws:lambda:us-east-1:468285055142:function:LiveISINPricingProd-LowIssuersLoader',
#             InvocationType='Event', Payload=json.dumps(payload))
#     elif (trigger_name == "general_liquidity") or (trigger_name == "issuer_liquidity"):
#         num_parts = num_part = np.round(len(ha_dict_invk) ** (1 / 3))
#         partitions = np.array_split(high_issuer_lst, num_parts)
#         print("sending partitions for Low issuers")
#         client = boto3.client('lambda')
#         for part in partitions:
#             part_dict = {}
#             for m_id in part.tolist():
#                 part_dict[m_id] = ha_dict_invk[m_id]
#             payload = {'high_issuers': part_dict, 'currency': [currency]}
#             # print(model_id)
#             print(payload)
#             response = client.invoke(
#                 FunctionName='arn:aws:lambda:us-east-1:468285055142:function:LiveISINPricingProd-LowIssuersLoader',
#                 InvocationType='Event', Payload=json.dumps(payload))


def lambda_handler(event, context):
    # print(event)
    if "Records" in event:
        for data in event['Records']:
            if data["eventName"] == "INSERT" or data["eventName"] == "MODIFY":

                if 'general_liquidity' in data['eventSourceARN']:
                    # print("triggered by general liquidties")
                    currency = data['dynamodb']['NewImage']['currency']['S']
                    ha_dict_invk = common_trigger(event, context,
                                                  all_issuers=False, trigger_name="general_liquidity", invoked=False,
                                                  currency=currency)
                    # if len(ha_dict_invk) >= 1:
                    #     inokeFunc(trigger_name="general_liquidity", ha_dict_invk=ha_dict_invk, currency=currency)
                    #     print("invoke")
                    # break
                    break


    else:
        # print("triggered by benchmark")
        model_bench = event['model_id']
        currency = model_bench[0][-3:].upper()
        ha_dict_invk = common_trigger(event, context, all_issuers=False, trigger_name="benchmark_models", invoked=True,
                                      currency=currency)
        # if len(ha_dict_invk) >= 1:
        #     inokeFunc(trigger_name="benchmark_models", ha_dict_invk=ha_dict_invk, currency=currency)
        #     print("invoke")
