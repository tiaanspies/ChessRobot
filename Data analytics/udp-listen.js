/*
Nodejs code to receive UDP packets from a Motive NatNet Server with Parsing
Lightweight -- ignores commands; just receives data and tracks ridgidboides
References PacketClient.cpp provided by NatNet SDK
Emily Lam 2019-06-28
*/

// Required modules
const dgram = require('dgram');           // UDP Datagrams
const process = require("process");       // Node.js process

// IPs and ports -- match information on Motive GUI
var PORT_MOTIVE = 1511;                   // Motive NatNet data channel
var LOCAL = '0.0.0.0';                    // All IPs on local machine
const MULTICAST_ADDR = "239.255.42.99";   // Motive NatNet multicast IP

// NATNET message ids
const NAT_FRAMEOFDATA = 7;                // ID for data frame

// IPs and ports -- ESP
var PORT_ESP = 3333;
var HOST =  '127.0.0.1';

// NATNET 3.1 compatiable with Motive 2.1.0 -- Tested July 9th, 2019
var major = 3;
var minor = 1;

// Define rigidbodies object
var rigidBodies = {};

// CONSTANTS
// IPs and ports to Python manager
var PORT_PYTHON_REC = 5000;
var PORT_PYTHON_SEND = 5001;
var PORT_PYTHON_LOG = 5002;
var MDATA = parseInt("0x44",16);
var MPOSREQ = parseInt("0x12", 16);

// Create a UDP socket and bind -- For Optitrack/Motive/NatNet
const socketOpti = dgram.createSocket({ type: "udp4", reuseAddr: true });
socketOpti.bind({address: LOCAL, port: PORT_MOTIVE}, function() {
  socketOpti.addMembership(MULTICAST_ADDR);
});

// Logs >> UDP socket listening on 0.0.0.0:1511 pid: XXXX
socketOpti.on("listening", function() {
  const address = socketOpti.address();
  console.log(
    `UDP socket listening on ${address.address}:${address.port} pid: ${
      process.pid
    }`
  );
});

// Handles errors
socketOpti.on('error', (err) => {
  console.error(`UDP error: ${err.stack}`);
});

// Handles messages
socketOpti.on('message', (msg, rinfo) => {
  console.log('Recieved UDP message ---------------------------');
  // console.log(msg);
  parseData(msg);
});

// Create a UDP socket and bind -- For python game manager
const socketPython = dgram.createSocket({ type: "udp4", reuseAddr: true });
socketPython.bind({address: LOCAL, port: PORT_PYTHON_REC});

// Logs >> UDP socket(Python Man) listening on 0.0.0.0:1511 pid: XXXX
socketPython.on("listening", function() {
  const address = socketPython.address();
  console.log(
    `UDP socket(PythonMan) listening on ${address.address}:${address.port} pid: ${
      process.pid
    }`
  );
});

// Handles errors
socketPython.on('error', (err) => {
  console.error(`UDP error: ${err.stack}`);
});

// Handles messages
socketPython.on('message', (msg, rinfo) => {
  // console.log('Recieved UDP message ---------------------------');
  // console.log(msg);
  parseDataPython(msg, rinfo);
});

function parseDataPython(msg, rinfo) {
  // Msg buffer index
  var offset = 0;

  //read Head
  var msgID = msg.readUIntLE(offset, 1); offset += 1;

  if(msgID==MPOSREQ)//Requested bot position
  { 
    var espID = msg.readUIntLE(offset, 1); offset += 1;
    console.log("Bot position requested from Python "+rinfo.address+", sending...");

    // Check if valid ID was requested
    if (rigidBodies['id'+espID] == undefined) {
      const payBuf = Buffer.from([MNODATA,espID]);
      console.log("Invalid ID requested ...");
      socketPython.send(payBuf, PORT_PYTHON_SEND, LOCAL,function(error){
        if(error){ console.log('MEH! It did not work!'); }
      });
      return;
    }

    const position = new Float32Array(3);             // Create pos array
    const orientation = new Float32Array(4);          // Create ori array

    // Populate position and orientation arrays
    position[0] = rigidBodies['id'+espID].position[0];
    position[1] = rigidBodies['id'+espID].position[1];
    position[2] = rigidBodies['id'+espID].position[2];
    orientation[0] = rigidBodies['id'+espID].orientation[0];
    orientation[1] = rigidBodies['id'+espID].orientation[1];
    orientation[2] = rigidBodies['id'+espID].orientation[2];
    orientation[3] = rigidBodies['id'+espID].orientation[3];
    
    const bufPos = Buffer.from(position.buffer,'hex');
    const bufOri = Buffer.from(orientation.buffer,'hex');
    const isTracked = Buffer.from([rigidBodies['id'+espID].tracked],'hex');

    // Data buffers
    const bufStr = Buffer.from([MDATA,espID],'hex');
    
    // Fill buffer and add checksum
    var payBuf = Buffer.concat([bufStr,bufPos,bufOri,isTracked]);
    console.log("Now, sending pos");
    socketPython.send(payBuf, PORT_PYTHON_SEND, LOCAL, function(error){
      if(error){ console.log('MEH! It did not work!'); }
    });
  }
  else{
    console.log("Unknown message head received");
  }
}
// Functions //////////////////////////////////////////////////////////////////
function parseData(msg) {
  // Msg buffer index
  var offset = 0;

  // First 2 Bytes is message ID
  var msgID = msg.readUIntLE(offset, 2); offset += 2;
  // console.log('Message: ' + msgID.toString(8));

  // Second 2 Bytes is the size of the packet
  var nBytes = msg.readUIntLE(offset, 2); offset += 2;
  // console.log('Bytes received: ' + nBytes.toString(8));

  // If FRAME OF MOCAP DATA packet (there are other message IDs)
  if (msgID == NAT_FRAMEOFDATA) {
    // console.log('>> Parsing data frame');

    // Next 4 Bytes is the frame number
    var frameNumber = msg.readUInt32LE(offset, offset += 4);
    // console.log('Frame: ' + frameNumber);

    // -----

    // Markersets (ignored) -----
    var nMarkerSets = msg.readUInt32LE(offset, offset += 4);
    // console.log('Markersets: ' + nMarkerSets);

    // Unlabeled markersets (ignored) -----
    var nOtherMarkerSets = msg.readUInt32LE(offset, offset += 4);
    // console.log('OtherMarkersets: ' + nOtherMarkerSets);

    // Rigid bodies -----

    // Next 4 Bytes is the number of rigidbodies
    var nRigidBodies = msg.readUInt32LE(offset, offset += 4);
    // console.log('RigidBodies: ' + nRigidBodies);

    var j;
    for (j = 0; j < nRigidBodies; j++) {
      // Rigid body ID
      var rID = msg.readUInt32LE(offset, offset += 4);
      // console.log('Rigidbody ID: ' + rID);

      // Rigid body position
      var x = msg.readFloatLE(offset, offset += 4);
      var y = msg.readFloatLE(offset, offset += 4);
      var z = msg.readFloatLE(offset, offset += 4);
      // console.log('-- Pos (mm):' + x*1000 + ',' + y*1000 + ',' + z*1000);

      // Rigid body orientation
      var qx = msg.readFloatLE(offset, offset += 4);
      var qy = msg.readFloatLE(offset, offset += 4);
      var qz = msg.readFloatLE(offset, offset += 4);
      var qw = msg.readFloatLE(offset, offset += 4);
      // console.log('-- Ori:' + qx + ',' + qy + ',' + qz + ',' + qw);

      // NatNet version 2.0 and later
      if(major >= 2) {
        // Mean marker error
        var fError = msg.readFloatLE(offset, offset += 4);
        // console.log('-- Mean marker error (mm): ' + fError*1000);
      }

      // NatNet version 2.6 and later
      if( ((major == 2)&&(minor >= 6)) || (major > 2) || (major == 0) ) {
        // params
        var params = msg.readUIntLE(offset, 2); offset += 2;
        var bTrackingValid = params & 0x01; // 0x01 : rigid body was successfully tracked in this frame
        // if (bTrackingValid == 0x01) { console.log('-- Rigidbody tracked') }
        // else { console.log('-- Rigidbody untracked')}
      }

      // Add/update rigidBodies object
      var rb = {
        id: rID,
        position: [x,y,z],
        orientation: [qx,qy,qz,qw],
        tracked: bTrackingValid
      };
      rigidBodies['id' + rID]  = rb;

    }

    // Skeletons (ignored) -----

    if( ((major == 2)&&(minor>0)) || (major>2)) {
      var nSkeletons = msg.readUInt32LE(offset, offset += 4);
      // console.log('Skeletons: ' + nSkeletons);
    }

    // Labeled Markers (ignored) -----
    // labeled markers (NatNet version 2.3 and later)
    // labeled markers - this includes all markers: Active, Passive, and 'unlabeled' (markers with no asset but a PointCloud ID)

    if( ((major == 2)&&(minor>=3)) || (major>2)) {
      var nLabeledMarkers = msg.readUInt32LE(offset, offset += 4);
      // console.log('Labeled Markers: ' + nLabeledMarkers);
    }

    // Force Plates (ignored) -----
    // Force Plate data (NatNet version 2.9 and later)
    if (((major == 2) && (minor >= 9)) || (major > 2)) {
      var nForcePlates = msg.readUInt32LE(offset, offset += 4);
      // console.log('Force Plates: ' + nForcePlates);
    }

    // Devices (ignored) -----
    // Device data (NatNet version 3.0 and later)
    if (((major == 2) && (minor >= 11)) || (major > 2)) {
      var nDevices = msg.readUInt32LE(offset, offset += 4);
      // console.log('Devices: ' + nDevices);
    }

    // -----

    // Timecode (ignored)
		var timecode = msg.readUInt8(offset, offset += 4);
    var timecodeSub = msg.readUInt8(offset, offset += 4)

    // Timestamp
    // NatNet version 2.7 and later - increased from single to double precision
    if( ((major == 2)&&(minor>=7)) || (major>2)) {
      var timestamp = msg.readDoubleLE(offset, offset += 4);
    }
    else {
      var timestamp = msg.readFloatLE(offset, offset += 4);
    }
    // console.log('Timestamp: ' + timestamp)

    // High res timestamps (version 3.0 and later) -- completely ignored
    if ( (major >= 3) || (major == 0) ) { offset += 24; }

    // Frame params (ignored)
    var params = msg.readUIntLE(offset, 2); offset += 2;
    var bIsRecording = params & 0x01; // 0x01 : Motive is recording
    // if (bIsRecording == 0x01) { console.log('Motive recording')}
    // else { console.log('Motive not recording')}
    var bTrackedModelsChanged = params & 0x02; // 0x02 Act trckd mdl list chngd
    // if (bTrackedModelsChanged  == 0x02) { console.log('Track list changed.')}
    // else { console.log('Track list unchanged')}

		// End of data tag
    var eod = msg.readInt8(offset, offset += 4);
    // if (eod == 0x00) {
    //   console.log('End; bytes read: ' + offset);
    //   console.log('------------------------------------------------')
    // }
  }

}