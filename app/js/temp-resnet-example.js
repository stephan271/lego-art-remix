async function runExample() {
    // Create an ONNX inference session with WebGL backend.
    console.log("hello from runExample");
    // const session = new onnx.InferenceSession({backendHint: "webgl"});

    const session = new onnx.InferenceSession({backendHint: "cpu"});
    // Load an ONNX model
    console.log("loading model");

    await session.loadModel("models/model-small.onnx");
    console.log("loading model");

    // Load image.
    // const imageLoader = new ImageLoader(imageSize, imageSize);
    // const imageData = await imageLoader.getImageData("temp/resnet-cat.jpg");
    const imageData = getPixelArrayFromCanvas(
        document.getElementById("step-1-canvas")
    );

    // Preprocess the image data to match input dimension requirement, which is 1*3*256*256.
    const width = 256;
    const height = 256;
    console.log("Preprocessing data");

    console.log({imageData});
    const preprocessedData = preprocess(imageData, width, height);

    console.log("constructing inputTensor");
    const inputTensor = new onnx.Tensor(preprocessedData, "float32", [
        1,
        3,
        width,
        height
    ]);

    console.log(inputTensor)

    console.log("running model");

    // Run model with Tensor inputs and get the result.
    const outputMap = await session.run([inputTensor]);
    console.log("extracting output data");

    const outputData = outputMap.values().next().value.data;

    console.log({outputData});
    const maxHeight = Math.max(...outputData);
    console.log(maxHeight);
    const normalizedOutputData = outputData.map(val=>Math.floor(255*val/maxHeight));
    console.log(normalizedOutputData)



    // const resultImageData = Uint8ClampedArray.from();
    const result = [];
    for (let i = 0; i < 256*256;i+=1) {
      for (let j = 0; j < 1;j++) {
        result.push(255-normalizedOutputData[i]);
      }
      result.push(255);
    }
    console.log({result})

     document.getElementById("step-2-canvas-upscaled").width = 256;
     document.getElementById("step-2-canvas-upscaled").height =256;
    // step2CanvasUpscaledContext.imageSmoothingEnabled = false;
    drawPixelsOnCanvas(result, document.getElementById("step-2-canvas-upscaled"))
}

/**
 * Preprocess raw image data to match Resnet50 requirement.
 */
function preprocess(data, width, height) {
    console.log("hello from preprocess");

    console.log({data})
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    console.log({dataFromImage})
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [
        1,
        3,
        height,
        width
    ]);
    console.log({dataProcessed})


    // Normalize 0-255 to (-1)-1
    ndarray.ops.divseq(dataFromImage, 128.0);
    ndarray.ops.subseq(dataFromImage, 1.0);

    // Realign imageData from [256*256*4] to the correct dimension [1*3*256*256].
    ndarray.ops.assign(
        dataProcessed.pick(0, 0, null, null),
        dataFromImage.pick(null, null, 2)
    );
    ndarray.ops.assign(
        dataProcessed.pick(0, 1, null, null),
        dataFromImage.pick(null, null, 1)
    );
    ndarray.ops.assign(
        dataProcessed.pick(0, 2, null, null),
        dataFromImage.pick(null, null, 0)
    );

    return dataProcessed.data;
}
