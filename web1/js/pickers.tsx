
import * as React from 'react';
import {useState, useEffect} from 'react';
import * as ReactDOM from 'react-dom';
import {DropdownButton} from './dropdown_button';
import * as _ from 'lodash';

async function jobItems(jobType, filterFn, wsId) {
    const resp = await fetch(`/api/list_jobs/?ws=${wsId}&job_type=${jobType}`);
    const respData = await resp.json();
    const jobs = respData.data;

    const goodJobs = jobs.filter(filterFn);

    return goodJobs.map(x => ({
        label: x.label,
        data: x,
        wsId: wsId,
        jobId: x.id,
    });
}

async function wfJobItems(wsId, subJobs) {
    const resp = await fetch(`/api/list_jobs/?ws=${wsId}&job_type=wf`);
    const respData = await resp.json();
    const jobs = respData.data;
    const items = []
    jobs.forEach(x => {
        const wzsJobs = []

        const inputJids = x.all_input_jids;
        // Weights will be an array for each score.  Pathways should only have
        // 1 weight, but prot scores can have multiple because they have multiple downstream
        // outputs that get weighted separately.
        const weights = x.input_weights

        for (const subJob of subJobs) {
            if (inputJids.indexOf(parseInt(subJob.jobId)) != -1) {
                let subJobData = {...subJob}
                if (subJob.jobId in weights) {
                    subJobData.weights = weights[subJob.jobId];
                }
                subJobData.groupLabel = x.label;
                wzsJobs.push(subJobData);
            }
        }
        if (wzsJobs.length == 0) return;

        items.push({
            label: `${x.label} (${wzsJobs.length} jobs)`,
            data: wzsJobs,
            scoretype: 'joblist',
            weights: weights,
        });
    });
    return items;
}

async function inputJobItems(jobType, wsId) {
    const resp = await fetch(`/api/list_jobs/?ws=${wsId}&job_type=${jobType}`);
    const respData = await resp.json();
    const jobs = respData.data;
    return jobs.map(x => {
        const input = x.parms['input_score'];
        const [jid, code] = input.split('_');
        const inputLabelParts = x.label.split('_');
        inputLabelParts.pop();
        return {
            label: inputLabelParts.join('_'),
            data: x,
            wsId: wsId,
            jobId: jid,
            code: code,
        }
    });
}

async function workspaceItems(nextItemsFn, allWs) {
    const resp = await fetch(`/api/list_workspaces/?active_only=${!allWs}`);
    const respData = await resp.json();
    const wses = respData.data;

    const wsOptions = wses.map((x) => ({
        'label': x.name,
        'items': nextItemsFn.bind(this, x.id),
    }));

    if (!allWs) {
        wsOptions.push({
            'label': 'All Workspaces',
            items: workspaceItems.bind(this, nextItemsFn, true),
        }
    }
    return wsOptions;
}


export const GlfPicker = ({onSelected}) => {
    const jobFilterFn = (job) => {
        // We used to filter to only new data, but all works now.
        // Going to leave this here just in case we need it in the future.
        //return job.parms['std_gene_list_set'].indexOf('.v1') != -1;
        return true;
    };
    const boundJobItems = async (wsId) => {
        const pathJobs = await jobItems('glf', jobFilterFn, wsId);
        let wfJobs = await wfJobItems(wsId, pathJobs);
        // We used to require that these have weights so that we could weight things, but
        // that filters out uniprot refreshes.
        // If there are no weights, the downstream code just gives everything weight 1.
        //wfJobs = wfJobs.filter((x) => !_.isEmpty(x.weights));
        return wfJobs.concat(pathJobs);
    }
    const wsItems = () => workspaceItems(boundJobItems, false);
    const classes = "btn btn-default";
    return (<DropdownButton
        text='Pathway Scores'
        items={wsItems}
        classNames={classes}
        onClick={item => onSelected(item)}
    />);
}

export const ProtScorePicker = ({onSelected}) => {
    const wzsItems = (wsId) => wzsJobs(wsId, 'codes')
    const boundJobItems = async (wsId) => {
        const protJobs = await inputJobItems('codes', wsId);
        const wzsJobs = await wfJobItems(wsId, protJobs);
        return wzsJobs.concat(protJobs);
    };
    const wsItems = () => workspaceItems(boundJobItems, false);
    const classes = "btn btn-default";
    return (<DropdownButton
        text='Protein Scores'
        items={wsItems}
        classNames={classes}
        onClick={item => onSelected(item)}
    />);
}

