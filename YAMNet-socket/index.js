import { WebSocketServer } from "ws";
import { getAudioClassPredictions} from "kromosynth";
import parseArgs from 'minimist';
const argv = parseArgs(process.argv.slice(2));
let port;
let host;
if( argv.hostInfoFilePath ) {
  // automatically assign port and write the info to the specified file path
  console.log("--- argv.hostInfoFilePath:", argv.hostInfoFilePath);
  port = 40051;
  argv.hostInfoFilePath.substring(argv.hostInfoFilePath.lastIndexOf("host-")+5).split("-").reverse().forEach( (i, idx) => port += parseInt(i) * (idx+1*10) );
  host = os.hostname();
  console.log("--- hostname:", host);
  fs.writeFile(argv.hostInfoFilePath, host, () => console.log(`Wrote hostname to ${argv.hostInfoFilePath}`));
} else {
  port = argv.port || process.env.PORT || '40051';
  host = "0.0.0.0";
}
const processTitle = argv.processTitle || 'kromosynth-evaluation-socket-server';
process.title = processTitle;
process.on('SIGINT', () => process.exit(1)); // so it can be stopped with Ctrl-C

const wss = new WebSocketServer({ host, port });

// TODO: read from parameters
const classificationGraphModel = "yamnet";
const modelUrl = "file:///Users/bjornpjo/Developer/vendor/tfjs-model_yamnet_tfjs_1/model.json";
const useGPU = true;

wss.on("connection", (ws) => {
  ws.on('error', console.error);
  ws.on("message", async (message) => {
    const buffer = new Uint8Array( message );
    const channelData = new Float32Array( buffer.buffer );
    const predictions = await getAudioClassPredictions(
      channelData,
      classificationGraphModel, modelUrl, useGPU
    );
    console.log(predictions);
    ws.send(JSON.stringify(predictions));
  });
});

console.log(`Evaluation WebSocketServer (YAMNet) listening on port ${port}`);