import { WebSocketServer } from "ws";
import parseArgs from 'minimist';
import os from "os";
import fs from "fs";
import hnswPkg from 'hnswlib-node';
const { HierarchicalNSW } = hnswPkg;
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
const processTitle = argv.processTitle || 'kromosynth-evaluation-socket-server: HNSW';
process.title = processTitle;
process.on('SIGINT', () => process.exit(1)); // so it can be stopped with Ctrl-C

const wss = new WebSocketServer({ host, port });

let modelUrl = argv.modelUrl;
// if modelUrl contains the string "localscratch/<job-ID>", replace the ID with the SLURM job ID
if( modelUrl && modelUrl.includes("localscratch") ) {
  // get the job-ID from from the environment variable SLURM_JOB_ID
  const jobId = process.env.SLURM_JOB_ID;
  console.log("Replacing localscratch/<job-ID> with localscratch/"+jobId+" in modelUrl");
  modelUrl = modelUrl.replace("localscratch/<job-ID>", `localscratch/${jobId}`);
}
console.log("modelUrl:",modelUrl);

// spaceName can be 'l2', 'ip, or 'cosine'
const spaceName = argv.spaceName || 'cosine';

// number of neighbors to return
let numNeighbors = argv.numNeighbors;

// modelUrl is the path to a folder containing the HNSW index file and a JSON file mapping indexes to keys
// - the index file is named 'hnswIndex.dat'
// - the indexToKey file is named 'indexToKey.json'
const indexPath = `${modelUrl}/hnswIndex.dat`;
const indexToKeyPath = `${modelUrl}/indexToKey.json`;
let index;
const indexToKey = JSON.parse( fs.readFileSync( indexToKeyPath, 'utf8' ) );

wss.on("connection", (ws) => {
  ws.on('error', console.error);
  ws.on("message", async (message) => {
    const messageParsed = JSON.parse(message);
    if( messageParsed.features ) {
      const startTime = performance.now();
      if( !index ) {
        const numDimensions = messageParsed.features.length
        index = new HierarchicalNSW(spaceName, numDimensions);
        index.readIndexSync( indexPath );
        if( !numNeighbors ) {
          numNeighbors = index.getCurrentCount();
        }
      }
      console.log(`Searching for ${numNeighbors} nearest neighbors to features of length ${messageParsed.features.length}`);
      const result = index.searchKnn( messageParsed.features, numNeighbors );
      const { neighbors, distances } = result;
      const taggedPredictions = {};
      neighbors.forEach( (neighbor, idx) => {
        // cosine distances:
        // 0 indicates no dissimilarity between vectors (they are the same).
        // 1 indicates that the vectors are orthogonal (uncorrelated).
        // 2 indicates that the vectors are diametrically opposed.
        // console.log(`distances[${idx}]:`, distances[idx]);
        // taggedPredictions[indexToKey[neighbor]] = 1 - (distances[idx] / 2); 
        taggedPredictions[indexToKey[neighbor]] = distances[idx]; 
      });
      // = neighbors.map( (neighbor, idx) => {
      //   return {
      //     [indexToKey[neighbor]]: 1 - distances[idx]
      //   }
      // } );
      const wsResponse = { taggedPredictions };
      const endTime = performance.now();
      console.log(`HNSW Prediction took ${endTime - startTime} ms`);
      ws.send(JSON.stringify(wsResponse));
    } else if( messageParsed.getKeys ) {
      ws.send(JSON.stringify(Object.values(indexToKey)));
    }
  });
});

console.log(`Evaluation WebSocketServer (HNSW) listening on port ${port}`);