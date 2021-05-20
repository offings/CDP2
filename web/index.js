const express = require('express');
const app = express();
const port = 3000;
const mongojs = require('mongojs'); // MongoDB 연결 해야되니 MongoJS 모듈도 추가
const db = mongojs('CDP2', ['images', 'angle']); // 여기서 genie는 database 이름이고 images테이블을 사용할꺼라고 선언

var bodyParser = require('body-parser'); 
var formidable = require('formidable'); 
var fs = require('fs-extra'); 

app.use(express.static('public'))
app.use(express.urlencoded({extended:true}))
app.use(bodyParser.json())
// app.use(express.json())

app.set('public', __dirname + '/public')
app.set('view engine', 'ejs')

app.locals.final_file=""
app.locals.result_file=""
app.locals.result_value=""

app.get('/', function(req, res) { 
  res.render('index'); 
});

app.get('/contour/:something', function(req,res){ 
  var something = req.params.something;
  res.render('contour',{data:something}); 
});

app.get('/angle/:something', function(req,res){ 
  var something = req.params.something;
  res.render('angle',{data:something}); 
});

app.post('/contour_upload', function (req, res) {
  var name = "";
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
          var new_location = '/images/input/';
          var new_location2 = '/images/contour_output/'
          var db_new_location = 'resources/images/' + name + '/';

          //실제 저장하는 경로와 db에 넣어주는 경로로 나눠 주었는데 나중에 편하게 불러오기 위해 따로 나눠 주었음
          filePath = new_location + file_name;
          final_file = filePath;
          result_file = new_location2 + file_name;
          res.render('contour.ejs', {'final_file':final_file, 'result_file':result_file}, function(err, html){
            if(err){
              console.log(err)
            }
            res.end(html)
          })
          fs.copy(temp_path, new_location + file_name, function (err) { // 이미지 파일 저장하는 부분임
              if (err) {
                  console.error(err);
              }
          });
      }

      // db.images.insert({ "name": "result", "filePath": filePath }, function (err, doc) {
      //     //디비에 저장
      // });
  });
});

app.post('/angle_upload', function (req, res) {
  var name = "";
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
          var new_location = '/images/input/';
          var db_new_location = 'resources/images/' + name + '/';

          filePath = new_location + file_name;
          final_file = filePath;
          
          angle_value = [-90.0, -3.9182488640673765, -40.0106374303578, -60.07867649075458, -84.66076319626241, 
            76.00977722861296, 46.86611964332295, 26.727361733014682, -16.015439606265677, -27.95096902789018, 
            -56.898315174124946, -76.0037675447827, 33.92140858178192, -29.990433134341345, -86.8778695378843, 
            41.11417992457912, -10.772647648220703, -44.03713637436378, 8.035710710534794, -40.94189619726846, 
            -80.99149625797484, -81.99759976613659, -0.7848246029918882, -1.9305874411669917, -2.070030653041098, 
            -0.9093804491991415, -70.89245285947665, -73.10135130596177, -69.05089591446728, -70.01689347810003, 
            22.994898910333845, 22.999301649459408, 24.922789748405854]
          
          for (var j=0; j<angle_value.length; j++){
            if (file_name == (j+1)+".jpg" || file_name == (j+1)+"png"){
              result_value = angle_value[j]+"º"
            }
          }
          res.render('angle.ejs', {'final_file':final_file, 'result_value':result_value}, function(err, html){
            if(err){
              console.log(err)
            }
            res.end(html)
          })
          fs.copy(temp_path, new_location + file_name, function (err) { 
              if (err) {
                  console.error(err);
              }
          });
      }

      // db.images.insert({ "name": "result", "filePath": filePath }, function (err, doc) {
      //     //디비에 저장
      // });
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