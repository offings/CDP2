const express = require('express');
const app = express();
const port = 3000;
const mongojs = require('mongojs'); // MongoDB 연결 해야되니 MongoJS 모듈도 추가
const db = mongojs('CDP2', ['images']); // 여기서 genie는 database 이름이고 images테이블을 사용할꺼라고 선언

var bodyParser = require('body-parser'); 
var formidable = require('formidable'); 
var fs = require('fs-extra'); 

app.use(express.static('public'))
app.use(express.urlencoded({extended:true}))
app.use(express.json())

app.set('public', __dirname + '/public')
app.set('view engine', 'ejs')


app.get('/', function(req, res) { 
  res.render('index'); 
});

app.get('/contour/:something', function(req,res){ 
  var something = req.params.something;
  res.render('contour',{data:something}); 
});

app.get('/circle/:something', function(req,res){ 
  var something = req.params.something;
  res.render('circle',{data:something}); 
});

app.get('/angle/:something', function(req,res){ 
  var something = req.params.something;
  res.render('angle',{data:something}); 
});

app.post('/upload', function (req, res) {
  var name = "";
  var filePath = "";
  var form = new formidable.IncomingForm();

  form.parse(req, function (err, fields, files) {
      name = fields.name;
  });

  form.on('end', function (fields, files) {
      for (var i = 0; i < this.openedFiles.length; i++) {
          var temp_path = this.openedFiles[i].path;
          var file_name = this.openedFiles[i].name;
          var index = file_name.indexOf('/');
          var new_file_name = file_name.substring(index + 1);
          var new_location = 'public/resources/images/' + name + '/';
          var db_new_location = 'resources/images/' + name + '/';

          //실제 저장하는 경로와 db에 넣어주는 경로로 나눠 주었는데 나중에 편하게 불러오기 위해 따로 나눠 주었음
          filePath = db_new_location + file_name;
          fs.copy(temp_path, new_location + file_name, function (err) { // 이미지 파일 저장하는 부분임
              if (err) {
                  console.error(err);
              }
          });
      }

      db.images.insert({ "name": name, "filePath": filePath }, function (err, doc) {
          //디비에 저장
      });
  });
});

app.get('/images', function(req, res){
  db.images.file_name(function(err, doc){
    res.json(doc)
  });
});

app.listen(port, function(){
  console.log('server on! http://localhost:'+port);
});