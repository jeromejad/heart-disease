var express = require("express");
var app = express();
var bodyParser = require("body-parser");
var router = express.Router();
var connect = require("connect");

var app = connect.createServer().use(connect.static(__dirname));


router.get("/", function(req, res){
	res.render("/index.html");
	
});

app.listen(process.env.PORT,process.env.IP, function(){
	console.log("hello");
});
		