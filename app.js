var express = require("express");
var app = express();
var bodyParser = require("body-parser");
var router = express.Router();
app.set("view engine", "html");
app.use(express.static(__dirname));
router.get("/", function(req, res){
	res.render("index");
	
});

app.listen(process.env.PORT,process.env.IP, function(){
	console.log("hello");
});
		