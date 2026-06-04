const ort = require("onnxruntime-node");
const fs = require("fs");

async function main() {
  const modelPath = "E:/Datascience2026/machinelearning/src/mlapp/models/adv_linear_model.onnx";

  const modelBuffer = fs.readFileSync(modelPath);
  const session = await ort.InferenceSession.create(modelBuffer);

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  // ✅ all session usage stays inside main()
  const single = new ort.Tensor("float32", Float32Array.from([150]), [1, 1]);
  const r1 = await session.run({ [inputName]: single });
  console.log(`TV spend 150 -> Sales: ${r1[outputName].data[0].toFixed(2)}`);

  const values = [50, 120, 230];
  const batch = new ort.Tensor("float32", Float32Array.from(values), [values.length, 1]);
  const r2 = await session.run({ [inputName]: batch });
  Array.from(r2[outputName].data).forEach((p, i) => {
    console.log(`  TV spend ${values[i]} -> Sales: ${p.toFixed(2)}`);
  });
}

main().catch((err) => console.error("Error:", err.message));