from pymongo import MongoClient
import gridfs

def mongo_conn() :
    try :
        conn = MongoClient(host='localhost', port=27017)
        print("MongoDB connected", conn)
        return conn.grid_file
    except Exception as e :
        print("Error in mongo connection", e)

db = mongo_conn()
name = "output.jpg"
file_location = ""+name
file_data = open(file_location, "rb")
data = file_data.read()
fs = gridfs.GridFS(db)
fs.put(data, filename=name)
print("upload complete")

data = db.fs.files.find_one({'filename' : name})
my_id = data['_id']
outputdata = fs.get(my_id).read()
download_location = "image/" + name
output = open(download_location, "wb")
output.write(outputdata)
output.close()
print("download complete")