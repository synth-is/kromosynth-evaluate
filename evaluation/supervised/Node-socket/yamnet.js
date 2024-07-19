import { WebSocketServer } from "ws";
import { getAudioClassPredictions} from "kromosynth";
import parseArgs from 'minimist';
import os from "os";
import fs from "fs";
const argv = parseArgs(process.argv.slice(2));
let port;
let host;
if( argv.hostInfoFilePath ) {
  // automatically assign port and write the info to the specified file path
  console.log("--- argv.hostInfoFilePath:", argv.hostInfoFilePath);
  port = 40051;
  argv.hostInfoFilePath.substring(argv.hostInfoFilePath.lastIndexOf("host-")+5).split("-").reverse().forEach( (i, idx) => port += parseInt(i) * (idx+1*10) );
  host = os.hostname();
  const hostname = `${host}:${port}`;
  console.log("--- hostname:", hostname);
  fs.writeFile(argv.hostInfoFilePath, hostname, () => console.log(`Wrote hostname to ${argv.hostInfoFilePath}`));
} else {
  port = argv.port || process.env.PORT || '40051';
  host = "0.0.0.0";
}
const processTitle = argv.processTitle || 'kromosynth-evaluation-socket-server: YAMNet';
process.title = processTitle;
process.on('SIGINT', () => process.exit(1)); // so it can be stopped with Ctrl-C

const wss = new WebSocketServer({ host, port });

// TODO: read from parameters
const classificationGraphModel = "yamnet";
console.log("classificationGraphModel:",classificationGraphModel);
let modelUrl = argv.modelUrl;
// if modelUrl contains the string "localscratch/<job-ID>", replace the ID with the SLURM job ID
if( modelUrl && modelUrl.includes("localscratch") ) {
  // get the job-ID from from the environment variable SLURM_JOB_ID
  const jobId = process.env.SLURM_JOB_ID;
  console.log("Replacing localscratch/<job-ID> with localscratch/"+jobId+" in modelUrl");
  modelUrl = modelUrl.replace("localscratch/<job-ID>", `localscratch/${jobId}`);
}
console.log("modelUrl:",modelUrl);
const useGPU = argv.useGPU;;
console.log("useGPU:",useGPU);

wss.on("connection", (ws) => {
  ws.on('error', console.error);
  ws.on("message", async (message) => {
    const buffer = new Uint8Array( message );
    const channelData = new Float32Array( buffer.buffer );
    console.log(`Predicting audio classes from  ${channelData.length} samples, classificationGraphModel: ${classificationGraphModel}, modelUrl: ${modelUrl}, useGPU: ${useGPU}`);
    const predictions = await getAudioClassPredictions(
      channelData,
      classificationGraphModel, modelUrl, useGPU
    );
    // prepend the keys in predictions.taggedPredictions with 'YAM_:'
    predictions.taggedPredictions = Object.keys(predictions.taggedPredictions).reduce( (acc, key) => {
      acc[`YAM_${key}`] = predictions.taggedPredictions[key];
      return acc;
    }, {});
    // console.log("predictions:", predictions);
    ws.send(JSON.stringify(predictions));
  });
});

console.log(`Evaluation WebSocketServer (YAMNet) listening on port ${port}`);