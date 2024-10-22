import { WebSocketServer } from "ws";
import parseArgs from 'minimist';
import os from "os";
import fs from "fs";
import hnswPkg from 'hnswlib-node';
import { parse as parseUrl } from 'url';
const { HierarchicalNSW } = hnswPkg;
import crypto from 'crypto';
import net from 'net';

function calculateL2Norm(vector) {
    return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
}

function normalizeVector(vector) {
    const norm = calculateL2Norm(vector);
    // Only add epsilon if norm is very close to zero
    const adjustedNorm = norm < 1e-8 ? 1e-8 : norm;
    return vector.map(val => val / adjustedNorm);
}

function improvedCosineDistance(vec1, vec2) {
    const normVec1 = normalizeVector(vec1);
    const normVec2 = normalizeVector(vec2);
    
    // Calculate dot product
    const dotProduct = normVec1.reduce((sum, val, i) => sum + val * normVec2[i], 0);
    
    // Clip to [-1, 1] range
    const similarity = Math.max(-1, Math.min(1, dotProduct));
    
    // More aggressive scaling for MFCC features:
    // - Map [-1, 1] to [0, 1] with stronger penalty for dissimilarity
    // - Apply power transformation to push down lower similarities
    const normalizedSim = (similarity + 1) / 2;
    return Math.pow(normalizedSim, 3);  // Cubic transformation to penalize dissimilarity
}

function euclideanDistance(vec1, vec2) {
    // For MFCC features, we might want to work with raw differences
    const distance = Math.sqrt(vec1.reduce((sum, val, i) => {
        const diff = val - vec2[i];
        return sum + diff * diff;
    }, 0));
    
    // More aggressive scaling for early evolution
    const maxPossibleDistance = Math.sqrt(vec1.length * 4);  // Assuming values in [-1,1]
    const similarity = 1 - (distance / maxPossibleDistance);
    return Math.pow(Math.max(0, similarity), 2);  // Square to penalize differences
}

function adaptiveSimilarity(vec1, vec2, metric) {
    const dim = vec1.length;
    let similarity;
    
    if (metric === 'l2' || dim <= 3) {
        similarity = euclideanDistance(vec1, vec2);
    } else {
        similarity = improvedCosineDistance(vec1, vec2);
        // Apply additional power transformation based on dimensionality
        if (dim <= 50) {
            similarity = Math.pow(similarity, 2);  // More aggressive for early evolution
        } else {
            similarity = Math.pow(similarity, 3);  // Even more aggressive for high-dimensional MFCC
        }
    }
    
    return Math.max(0, Math.min(1, similarity));
}

function applyPowerTransformation(similarity, power) {
    return Math.pow(similarity, power);
}

const argv = parseArgs(process.argv.slice(2));
let port;
let host;
if (argv.hostInfoFilePath) {
//   port = 40051;
//   argv.hostInfoFilePath.substring(argv.hostInfoFilePath.lastIndexOf("host-")+5).split("-").reverse().forEach((i, idx) => port += parseInt(i) * (idx+1*10));
  let hostInfoFilePath;
  if( process.env.pm_id ) { // being managed by PM2
    hostInfoFilePath = `${argv.hostInfoFilePath}${parseInt(process.env.pm_id) + 1}`;
  } else {
    hostInfoFilePath = argv.hostInfoFilePath;
  }
  port = await filepathToPort( hostInfoFilePath );
  host = os.hostname();
  const hostname = `${host}:${port}`;
  fs.writeFile(hostInfoFilePath, hostname, () => console.log(`Wrote hostname to ${hostInfoFilePath}`));
} else {
  port = argv.port || process.env.PORT || '40051';
  host = "0.0.0.0";
}

const processTitle = argv.processTitle || 'kromosynth-evaluation-socket-server: HNSW';
process.title = processTitle;
process.on('SIGINT', () => process.exit(1));

const wss = new WebSocketServer({ host, port });

let modelUrl = argv.modelUrl;
if (modelUrl && modelUrl.includes("localscratch")) {
  const jobId = process.env.SLURM_JOB_ID;
  modelUrl = modelUrl.replace("localscratch/<job-ID>", `localscratch/${jobId}`);
}

const spaceName = argv.spaceName || 'cosine';
const indexPath = `${modelUrl}/hnswIndex.dat`;
const indexToKeyPath = `${modelUrl}/indexToKey.json`;
let index;
const indexToKey = JSON.parse(fs.readFileSync(indexToKeyPath, 'utf8'));

wss.on("connection", (ws, req) => {
  ws.on('error', console.error);
  
  ws.on("message", async (message) => {
      const startTime = performance.now();
      const messageParsed = JSON.parse(message);
      
      if (messageParsed) {
          const urlParams = parseUrl(req.url, true).query;
          const k = parseInt(urlParams.k) || 1;
          const useAdaptiveSimilarity = urlParams.useAdaptive !== 'false';  // Default to true
          const powerTransform = parseFloat(urlParams.power) || 1; // Default to 1 (no additional transformation)
          
          if (!index) {
              const numDimensions = messageParsed.length;
              index = new HierarchicalNSW(spaceName, numDimensions);
              index.readIndexSync(indexPath);
          }
          
          const result = index.searchKnn(messageParsed, k);
          const { neighbors, distances } = result;

          let printNeighbourKeysAndDistances = true; // TODO: configurable
            if (printNeighbourKeysAndDistances && neighbors.length > 0) {
              const key = indexToKey[neighbors[0]];
              console.log(`Neighbor 0: key=${key}, distance=${distances[0]}`);
            }
          
          let similarities;
          if (useAdaptiveSimilarity) {
              similarities = neighbors.map(neighbor => {
                  const referenceFeatures = index.getPoint(neighbor);
                  return adaptiveSimilarity(messageParsed, referenceFeatures, spaceName);
              });
          } else {
              // Convert raw distances to similarities
              similarities = distances.map(distance => 1 / (1 + distance));
          }
          
          // Apply additional power transformation
          const transformedSimilarities = similarities.map(sim => applyPowerTransformation(sim, powerTransform));
          
          // Use geometric mean for aggregation
          const geometricMean = Math.exp(
              transformedSimilarities.reduce((sum, val) => sum + Math.log(val + 1e-8), 0) / transformedSimilarities.length
          );
          
          const wsResponse = {
              fitness: geometricMean
          };
          
          ws.send(JSON.stringify(wsResponse));
      }
  });
});

console.log(`Evaluation WebSocketServer (HNSW) listening on port ${port}`);

function isPortTaken(port) {
  return new Promise((resolve) => {
      const server = net.createServer()
          .once('error', () => resolve(true))
          .once('listening', () => server.once('close', () => resolve(false)).close())
          .listen(port);
  });
}
async function filepathToPort(filepath, variation = 0) {
    let filepathVariation = filepath + variation.toString();
    let hash = crypto.createHash('md5').update(filepathVariation).digest("hex");
    let shortHash = parseInt(hash.substring(0, 8), 16);
    let port = 1024 + shortHash % (65535 - 1024);
    let isTaken = await isPortTaken(port);
  
    if(isTaken) {
        console.log(`--- filepathToPort(${filepath}): port ${port} taken`)
        return await filepathToPort(filepath, variation + 1);
    } else {
        console.log(`--- filepathToPort(${filepath}): port ${port} available`);
        return port;
    }
  }