const save = require("./utils/saveFile");
const path = require("path"); 
const tf = require("@tensorflow/tfjs-node"); 
const canvas = require("canvas");
const faceapi = require("@vladmandic/face-api/dist/face-api.node.js");

const modelPathRoot = "./models";  
const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

 
//main function
async function main(file) { 
//brepare face api 
  console.log("FaceAPI single-process test"); 
  await faceapi.tf.setBackend("tensorflow");
  await faceapi.tf.enableProdMode();
  await faceapi.tf.ENV.set("DEBUG", false);
  await faceapi.tf.ready(); 
  console.log(
    `Version: TensorFlow/JS ${faceapi.tf?.version_core} FaceAPI ${
      faceapi.version.faceapi
    } Backend: ${faceapi.tf?.getBackend()}`
  );

  //lode model from model file
  console.log("Loading FaceAPI models");
  const modelPath = path.join(__dirname, modelPathRoot); 
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);  
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath) 




  // call func return name
  return faceRecognition(file); 

 
}



let labeledFaceDescriptors; 
async function faceRecognition(imageFile) { 
 // Face recognion //////////////////////////////////////////////////////////////

  // get img we need recognize
  const referenceImage = await canvas.loadImage(imageFile) 
  // lode traning img handler function
  if(!labeledFaceDescriptors){
    labeledFaceDescriptors = await loadLabeledImages();  
   }  // 
  const faceMatcher = new faceapi.FaceMatcher(
    labeledFaceDescriptors,
    0.6
  );
    // get img with and hight
  const displaySize = { width: referenceImage.width, height: referenceImage.height };
  // detect all face and Descriptors of faces 
  const detections = await faceapi
    .detectAllFaces(referenceImage)
    .withFaceLandmarks()
    .withFaceDescriptors();
    // resize to compare Descriptor
  const resizedDetections = faceapi.resizeResults(
    detections,
    displaySize
  );
  // get best match descriptor
  const results = resizedDetections.map((d) =>
    faceMatcher.findBestMatch(d.descriptor)
  );

  console.log(results[0].label);


  // lope all faces an print all if detected or not
  results.forEach((result, i) => {
    const box = resizedDetections[i].detection.box;
    console.log(result.toString());
  });


  return results[0].label;
}  

 
// function handel traninig image for tran  model
function loadLabeledImages() { 
  const labels = [
    'Black Widow',
    'Captain America',
    'Captain Marvel',
    'Hawkeye',
    'Jim Rhodes',
    'Thor',
    'Tony Stark'
  ];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) { 
        const img =await canvas.loadImage(
          `http://localhost:3000/${label}/${i}.jpg`
        );
        console.log("kjhlkjkkkkkkkkkkkkkkkkkkkkkkkk");
        console.log(img);
        console.log(descriptions);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
 
module.exports = {
  recognize: main,
};

