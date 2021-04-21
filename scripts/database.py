import os
from os.path import join, dirname
import pymongo
import urllib
from dotenv import load_dotenv, find_dotenv
from scripts import hashing as ha

load_dotenv(find_dotenv())

print(os.environ.get("TEST"))


def connect_to_db():
    s1 = os.environ.get("MONGODBURL1")
    pwd = os.environ.get("MONGODBPWD")
    s2 = os.environ.get("MONGODBURL2")
    client = pymongo.MongoClient(s1 + urllib.parse.quote(pwd) + s2)
    db = client.jobySite
    return db


def add_data_to_db(fd):
    db = connect_to_db()
    col = db.users
    if not check_if_email_already_there(col, fd["email"]):
        di = fd
        di["pwd"] = ha.hash_password(di["pwd"])
        di["email"] = di["email"].lower()
        col.insert_one(di)
        return True
    return False


def check_if_email_already_there(col, email):
    x = col.find_one({"email": email.lower()})
    if x:
        return True
    else:
        return False


def find_user(loginDict):
    db = connect_to_db()
    col = db.users
    x = col.find_one({"email": loginDict["email"].lower()})
    print("X is", x)
    if x:
        if ha.verify_password(x["pwd"], loginDict["pwd"]):
            return True
        else:
            return False
    else:
        return False


def read_from_db():
    pass


def make_mongo_dict(fd):

    return {}


# add_data_to_db({"pwd": "sghgh", "email": "S@s.com"})
# find_user({"pwd": "sghgh", "email": "k@s.com"})
