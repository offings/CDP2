from pymongo import MongoClient
import gridfs

# def mongo_conn() :
#     try :
#         conn = MongoClient(host='localhost', port=27017)
#         print("MongoDB connected", conn)
#         return conn.grid_file
#     except Exception as e :
#         print("Error in mongo connection", e)

ip = 'localhost'
port = 27017

connection = MongoClient(ip, port)
print(connection.list_database_names())

db = connection.get_database('CDP2')
print(db.list_collection_names())

collection = db.get_collection('angle')

result = collection.find()
for i in result :
    print(i)

# Insert Data
collection.insert_one(
    {
        "name" : "angle_result",
        "angle" : 26.111,
    }
)

# name = "output.jpg"
# file_location = ""+name
# file_data = open(file_location, "rb")
# data = file_data.read()
# fs = gridfs.GridFS(db)
# fs.put(data, filename=name)
# print("upload complete")

# data = db.fs.files.find_one({'filename' : name})
# my_id = data['_id']
# outputdata = fs.get(my_id).read()
# download_location = "image/" + name
# output = open(download_location, "wb")
# output.write(outputdata)
# output.close()
# print("download complete")