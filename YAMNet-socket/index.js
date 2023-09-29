import { WebSocketServer } from "ws";
import { getAudioClassPredictions} from "kromosynth";

const wss = new WebSocketServer({ port: 9080 });

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
