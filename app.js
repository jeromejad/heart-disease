var express = require("express");
var app = express();
var bodyParser = require("body-parser");


router.get("/", function(req, res){
	res.render("index.html");
	
});

app.listen(process.env.PORT,process.env.IP, function(){
	console.log("hello");
});
		