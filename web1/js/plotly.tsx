
import * as React from 'react';
import {useState, useEffect} from 'react';
import * as ReactDOM from 'react-dom';

import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-cartesian-dist-min';
export const Plot = createPlotlyComponent(Plotly)


export const PlotlyPlot = ({apiUrl}) => {
    const [loading, setLoading] = useState(null);
    const [data, setData] = useState(null);
    if (apiUrl != loading) {
        setLoading(apiUrl);
        async function load() {
            const resp = await fetch(apiUrl);
            const data = await resp.json();
            setData(data);
        }
        load();
    }

    if (data) {
        return <Plot data={data.data} layout={data.layout} />
    } else {
        return null;
    }

}
