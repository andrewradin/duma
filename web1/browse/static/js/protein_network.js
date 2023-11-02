let g_wsId = "";
let g_ktProts = {};
function setupNetwork({wsId, ktProts}) {
    g_wsId = wsId;
    g_ktProts = {}
    for (const prot of ktProts) {
        g_ktProts[prot] = true;
    }
}



async function query({prots, depth, shownProts}) {
    if (g_wsId == "") {
        throw "Run setup first";
    }
    const curHref = `/${g_wsId}/proteinnetwork/`;

    const encoded = encodeURI(JSON.stringify({prots, depth, shownProts}));

    const resp = await fetch(`${curHref}?query=${encoded}`, {
        });
    const respData = await resp.json();
    return respData;
}
var cy = cytoscape({
  container: document.getElementById('cy'), // container to render in
  elements: [],
  style: [ // the stylesheet for the graph
      {
        selector: 'node',
        style: {
            shape: 'roundrectangle',
            label: 'data(label)',
            color: '#222',
            'background-color': '#f2faf4',
            'border-color': 'black',
            'border-width': '1.0px',
            'border-opacity': '0.2',
            'text-valign': 'center',
            'text-halign': 'center',
            width: '90px',
            height: '25px',
            }
          }, {
        selector: 'edge',
        style: {
          //label: 'data(label)',
          'width': 2,
          'line-color': '#ccc',
          'curve-style': 'straight',
          'target-arrow-color': '#c00',
          'target-arrow-shape': 'triangle-backcurve'
          }
        }, {
            selector: '[type="core"]',
            style: {
            'border-color': '#092',
            'border-width': '3.0px',
            'border-opacity': '1.0',
            }
        }, {
            selector: '[ktType="kt"]',
            style: {
            'background-color': '#fecda0',
            }
        }
    ],
    layout: {
        name: 'cose',
        }
  });
  
cy.on('click','node',function(event) {
    const node = event.target;
    addToNetwork([node.id()], 1);
});


let firstRun = true;

const allShownProts = []

async function addToNetwork(startProts, depth) {
    const newData = await query({prots: startProts, depth: depth, shownProts: allShownProts});
    const newCy = [];
    for (const protData of newData.prots) {
        const prot = protData['prot']
        const ktType = g_ktProts[prot] ? 'kt': '';
        if (ktType == 'kt') {
            console.info("Adding kt prot", prot)
        }
        newCy.push({
            group: 'nodes',
            data: {
                id: protData['prot'],
                label:protData['gene'],
                type: 'secondary',
                ktType,
            }
        });
        allShownProts.push(protData['prot']);
    }

    for (const edge of newData.edges) {
        const [p1, p2, ev, dir] = edge;
        newCy.push({group: 'edges', data: {
                id: `${p1}_${p2}`,
                label: ev,
                source: p1,
                target: p2,
        }});
    }

    const duration = firstRun ? 0 : 2000;
    firstRun = false;

    cy.add(newCy);

    for (const prot of startProts) {
        if (depth >= 1) {
            cy.$(`#${prot}`).data({'type': 'core'});
        }
    }

    cy.layout({
        name: 'cose-bilkent',
        idealEdgeLength: 300,
        animationDuration: duration,
        randomize: false,
    }).run();

}





