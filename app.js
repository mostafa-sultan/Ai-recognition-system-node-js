const express = require("express");
const fileUpload = require("express-fileupload");
const faceApiService = require("./faceApiService");
const faceRecognition = require("./faceRecognition");
const detectAllDataFace = require("./detectAllDataFace");

const path = require("path");

const app = express();
const port = process.env.PORT || 3000;

app.use(fileUpload());


app.use(express.static(path.join(__dirname, 'training_Image')));



app.post("/detectAllDataFace", async (req, res) => {
  const { file } = req.files;

  const result = await detectAllDataFace.detect(file.data, file.name);
   
  // var expressions=result[1][0].expressions;
  // console.log(result[0][0].label)
  // console.log(result[1][0].expressions)
  // console.log(result[1][0].gender)
  // console.log(result[1][0].age)

  res.json({
    name: result[0][0].label,
    gender: result[1][0].gender,
    age: result[1][0].age,
    Number_of_faces: result[1].length, 
    expresion: result[1][0].expressions,

    // detectedFaces: result,
    // url: `http://localhost:3000/out/${file.name}`,
  });
});







 

app.post("/upload", async (req, res) => {
  const { file } = req.files;

  const result = await faceApiService.detect(file.data, file.name);

  res.json({
    detectedFaces: result.length,
    url: `http://localhost:3000/out/${file.name}`,
  });
});




app.post("/recognize", async (req, res) => {
  const { file } = req.files;

  const result = await faceRecognition.recognize(file.data);

  res.json({
    name: result, 
  });
});

app.use("/out", express.static("out"));

app.listen(port, () => {
  console.log("Server started on port" + port);
});
