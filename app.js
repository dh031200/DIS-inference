let session;

async function onOpenCvReady() {
    console.log('OpenCV.js is ready.');
    // Load the ONNX model when the page is loaded
    try {
        session = await ort.InferenceSession.create('DIS-inference.onnx');
    } catch (error) {
        console.error('Failed to load the model:', error);
    }
}

window.onload = async function() {
    // Get references to the buttons
    const inferBtn = document.getElementById('inferBtn');
    const resetBtn = document.getElementById('resetBtn');
    const saveBtn = document.getElementById('saveBtn');

    document.getElementById('imageUpload').addEventListener('click', function () {
        document.getElementById('hiddenFileInput').click();
    });

    document.getElementById('hiddenFileInput').addEventListener('change', function (e) {
        const reader = new FileReader();
        reader.onload = function (event) {
            const img = new Image();
            img.onload = function () {
                document.getElementById('uploadedImage').src = event.target.result;
                // Enable the infer and reset buttons when an image is uploaded
                inferBtn.disabled = false;
                resetBtn.disabled = false;
            }
            img.src = event.target.result;
        }
        reader.readAsDataURL(e.target.files[0]);
    }, false);

    inferBtn.addEventListener('click', async function () {
        // Show the loading overlay
        document.getElementById('loadingOverlay').style.display = 'block';
        console.log('Loading overlay should be visible now');
        inferBtn.disabled = true;

        if (session) {
            setTimeout(async function () {
                try {
                    // Create an OpenCV Mat from the image
                    const img = cv.imread(document.getElementById('uploadedImage'));

                    // Pre-process the image
                    const resized = new cv.Mat();
                    const size = new cv.Size(1024, 1024);
                    cv.resize(img, resized, size, 0, 0, cv.INTER_LINEAR);

                    let tensorData = new Float32Array(3 * 1024 * 1024);
                    for (let y = 0; y < resized.rows; y++) {
                        for (let x = 0; x < resized.cols; x++) {
                            let pixel = resized.ucharPtr(y, x);
                            for (let c = 0; c < 3; c++) {
                                // normalize pixel data and arrange in [channel, height, width] format
                                tensorData[c * resized.rows * resized.cols + y * resized.cols + x] = pixel[2 - c] / 255.0 - 0.5;
                            }
                        }
                    }

                    const tensor = new ort.Tensor('float32', tensorData, [1, 3, resized.rows, resized.cols]);
                    const inputName = session.inputNames[0];
                    let inputs = {};
                    inputs[inputName] = tensor;
                    const outputMap = await session.run(inputs);
                    // session.run(inputs).then(outputMap => {
                    // Post-process
                    const outputName = session.outputNames[0];
                    const outputTensor = outputMap[outputName];
                    const output = new cv.Mat(1024, 1024, cv.CV_32FC1);
                    output.data32F.set(outputTensor.data);

                    const resizedOutput = new cv.Mat();
                    cv.resize(output, resizedOutput, img.size(), 0, 0, cv.INTER_LINEAR);

                    const minMax = cv.minMaxLoc(resizedOutput);
                    const _min = minMax.minVal;
                    const _max = minMax.maxVal;

                    const result = new cv.Mat();
                    let minMat = cv.Mat.ones(resizedOutput.rows, resizedOutput.cols, resizedOutput.type());
                    minMat = minMat.mul(minMat, _min);  // Multiply with _min
                    cv.subtract(resizedOutput, minMat, result);
                    minMat.delete();

                    const scale = 255 / (_max - _min);
                    let scaleMat = cv.Mat.ones(result.rows, result.cols, result.type());
                    scaleMat = scaleMat.mul(scaleMat, scale); // Multiply with scale
                    cv.multiply(result, scaleMat, result);
                    scaleMat.delete();

                    // Convert to Uint8 format
                    const uint8Result = new cv.Mat();
                    result.convertTo(uint8Result, cv.CV_8U);

                    // Create a 4-channel image from the original
                    const imgWithAlpha = new cv.Mat();
                    cv.cvtColor(img, imgWithAlpha, cv.COLOR_RGB2RGBA);

                    // Insert the result into the alpha channel of the new image
                    const imgData = imgWithAlpha.data;
                    const resultData = uint8Result.data;
                    for (let i = 0; i < img.rows * img.cols; i++) {
                        imgData[i * 4 + 3] = resultData[i];
                    }

                    // Hide the loading overlay
                    document.getElementById('loadingOverlay').style.display = 'none';
                    console.log('Loading overlay should be hidden now');

                    // Enable the save button when the inference is done
                    saveBtn.disabled = false;


                    // Display result
                    cv.imshow('result', imgWithAlpha);

                    // Hide the uploaded image
                    document.getElementById('uploadedImage').style.display = 'none';


                    // Clean up
                    img.delete();
                    resized.delete();
                    output.delete();
                    resizedOutput.delete();
                    result.delete();
                    uint8Result.delete();
                } catch (error) {
                    console.error('Failed to run the model:', error);
                }
            }, 0);
        } else {
            console.error('The model is not loaded yet.');
        }
    });

    resetBtn.addEventListener('click', function () {
        // Clear the uploaded image
        document.getElementById('hiddenFileInput').value = '';
        document.getElementById('uploadedImage').src = '';
        document.getElementById('uploadedImage').style.display = '';

        // Clear the result image
        const resultCanvas = document.getElementById('result');
        const ctx = resultCanvas.getContext('2d');
        ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

        // Disable the infer, reset, and save buttons
        inferBtn.disabled = true;
        resetBtn.disabled = true;
        saveBtn.disabled = true;
    });

    saveBtn.addEventListener('click', function () {
        const resultCanvas = document.getElementById('result');
        const link = document.createElement('a');
        link.download = 'result.png';
        link.href = resultCanvas.toDataURL();
        link.click();
    });
}