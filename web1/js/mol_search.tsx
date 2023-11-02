


import Autosuggest from 'react-autosuggest';
import * as React from 'react';
import * as ReactDOM from 'react-dom';
import {useState, useEffect} from 'react';
import {GeneProt} from './gene_prot';
import _ from 'lodash';

const theme = {
    container: {
        position: 'relative'
    },
    input: {
        width: '100%',
        height: 30,
        padding: '5px 5px',
        fontFamily: 'Helvetica, sans-serif',
        fontWeight: 300,
        fontSize: 16,
        border: '1px solid #aaa',
    },
    inputFocused: {
        outline: 'none'
    },
    inputOpen: {
        borderBottomLeftRadius: 0,
        borderBottomRightRadius: 0
    },
    suggestionsContainer: {
        display: 'none'
    },
    suggestionsContainerOpen: {
        display: 'block',
        position: 'absolute',
        top: 30,
        width: '100%',
        border: '1px solid #aaa',
        backgroundColor: '#fff',
        fontFamily: 'Helvetica, sans-serif',
        fontWeight: 300,
        fontSize: 16,
        borderBottomLeftRadius: 4,
        borderBottomRightRadius: 4,
        zIndex: 2
    },
    suggestionsList: {
        margin: 0,
        padding: 0,
        listStyleType: 'none',
    },
    suggestion: {
        cursor: 'pointer',
        padding: '10px 20px'
    },
    suggestionHighlighted: {
        backgroundColor: '#ddd'
    }
};

const loadingTheme = {
    ...theme,
    suggestionsContainerOpen: {
        ...theme.suggestionsContainerOpen,
        backgroundColor: '#f7f8f9',
    }
};

export const MolWithTargets = ({molData}) => {
    let targs = molData.targets.map((target) => (
        <GeneProt key={target[0]} gene={target[1]} prot={target[0]} direction={target[2]} />
    ));
    if (targs.length > 3) {
        targs = targs.slice(0, 3);
        targs.push(' ...');
    }
    return <span>{molData.name} {targs}</span>;
};

export const MolSearch = ({onSelected, wsId, fieldName, value, idToName}) => {
    if (idToName && idToName[value]) {
        // This is the multiinput init codepath, we get an id/name mapping and a wsa id.
        value = idToName[value];
    } else if (value && value.name) {
        // "Selected" value codepath, we have a 'suggestion' object.
        value = value.name;
    }

    const [stateValue, setValue] = useState(value || '');
    // We do some extra bookkeeping because requests often return out-of-order.
    // This is because the subsequent queries are narrower due to more letters.
    // We will only set a new suggestions if it's a more recent idx.
    const [suggestions, setSuggestions] = useState({
        reqIdx:0,
        setIdx:0,
        data: [],
    });

    const clearRequested = () => setSuggestions({...suggestions, data:[]});
    const fetchRequested = async ({value}) => {
        const myReq = suggestions.reqIdx + 1;
        setSuggestions({...suggestions, reqIdx: myReq});
        let url;
        if (!wsId) {
            url = `/api/search_drugs/${value}/`;
        } else {
            url = `/api/search_wsas/${wsId}/${value}/`;
        }
        const resp = await fetch(url);
        const respData = await resp.json();
        setSuggestions((prevState) => {
            if (prevState.setIdx <= myReq) {
                return {...prevState, setIdx: myReq, data: respData.data};
            } else {
                return prevState;
            }
        });

    };

    const renderSuggestion = (suggestion) => {
        return <MolWithTargets molData={suggestion} />
    };

    const selected = (evt, {suggestion}) => {
        onSelected(suggestion);
    };

    const inputProps = {
        placeholder: 'Molecule',
        value: stateValue,
        onChange: (evt, {newValue}) => setValue(newValue),
        name: fieldName,
    };
    let curTheme = theme;
    if (suggestions.reqIdx != suggestions.setIdx) {
        curTheme = loadingTheme;
    }
    return (
        <Autosuggest
            suggestions={suggestions.data}
            onSuggestionsFetchRequested={fetchRequested}
            onSuggestionsClearRequested={clearRequested}
            onSuggestionSelected={selected}
            getSuggestionValue={(suggestion) => suggestion.name}
            renderSuggestion={renderSuggestion}
            inputProps={inputProps}
            theme={curTheme}
        />
        );
};
MolSearch.formValueFromSelected = (selected) => {
    if (selected && selected.wsa_id) {
        return selected.wsa_id;
    } else {
        return selected;
    }
};


export const ProtSearch = ({value, onSelected}) => {
    const [stateValue, setValue] = useState(value || '');
    const [suggestions, setSuggestions] = useState([]);

    const clearRequested = () => setSuggestions([]);
    const fetchRequested = async ({value}) => {
        if (value.length >= 1) {
            const query = `search=${value}&limit=10&type=start`;
            const resp = await fetch(`/api/prot_search/?${query}`);
            const respData = await resp.json();
            setSuggestions(respData.matches);
        }
    };

    const renderSuggestion = (suggestion) => {
        const style = {
            backgroundColor: '#eef',
            border: '1px solid #bbb',
            fontSize: '75%',
            marginLeft: '3px',
            padding: '2px',
        };
        return (<span>
            <div>{suggestion.gene} ({suggestion.uniprot})</div>
            <div><small>{suggestion.name}</small></div>
        </span>);
    };

    const selected = (evt, {suggestion}) => {
        onSelected([suggestion.uniprot, suggestion.gene]);
    };

    const valueStr = (value) => {
        if (_.isString(value)) {
            return value;
        }
        const [prot, gene] = value;
        if (!prot) {
            return '';
        }
        return`${gene} (${prot})`;
    };

    const inputProps = {
        placeholder: 'Gene / Prot',
        value: valueStr(stateValue),
        onChange: (evt, {newValue}) => setValue(newValue),
    };
    return (
        <Autosuggest
            suggestions={suggestions}
            onSuggestionsFetchRequested={fetchRequested}
            onSuggestionsClearRequested={clearRequested}
            onSuggestionSelected={selected}
            getSuggestionValue={(suggestion) => [suggestion.uniprot, suggestion.gene]}
            renderSuggestion={renderSuggestion}
            inputProps={inputProps}
            theme={theme}
        />
        );
};

export const ProtAndDirSelect = ({value, onSelected}) => {
    const [stateValue, setValue] = useState(value || ['', '', 0]);
    const [uniprot, gene, dir] = stateValue;
    const onProtSelect = ([newUniprot, newGene]) => {
        setValue([newUniprot, newGene, dir]);
        onSelected([newUniprot, newGene, dir]);
    };
    const radioStyle = {
        marginLeft: '1rem',
        marginBottom: '2rem',
    }
    const onDirChange = (e) => {
        setValue([uniprot, gene, e.target.value]);
        onSelected([uniprot, gene, e.target.value]);
    };
    return (
        <div style={{'display': 'inline-block'}}>
            <ProtSearch value={[uniprot, gene]} onSelected={onProtSelect} />
            <input value={1} style={radioStyle} type='radio' checked={dir==1} onChange={onDirChange} /> Up
            <input value={-1} style={radioStyle} type='radio' checked={dir==-1} onChange={onDirChange} /> Down
            <input value={0} style={radioStyle} type='radio' checked={dir==0} onChange={onDirChange} /> Unknown
        </div>
    );
};


/** InputType must support:
 *  - Initial value of 'null' for a new entry
 *  - 'value' and 'onSelected' attributes
 */
export const MultiInput = ({InputType, initialEntries, onChange, extraAttrs}) => {
    const [entries, setEntries] = useState(initialEntries || []);
    extraAttrs = extraAttrs || {};

    const addEl = () => {
        setEntries([...entries, null]);
    }

    const onRemove = (idx) => {
        entries.splice(idx, 1);
        setEntries([...entries]);
        onChange(entries);
    }
    const onSelectedInternal = (idx, updatedEntry) => {
        entries[idx] = updatedEntry;
        setEntries([...entries]);
        onChange(entries);
    };
    const els = [];
    for (let i = 0; i < entries.length; ++i) {
        const entry = entries[i];
        els.push((
            <div key={[entry,i]}>
                <span style={{'display': 'inline-block'}}>
                <InputType value={entry} onSelected={onSelectedInternal.bind(this, i)} {...extraAttrs} />
                </span>
                <button type='button' style={{'verticalAlign': 'top', 'marginLeft': '0.5rem'}} className='btn btn-default btn-xs glyphicon glyphicon-trash' onClick={onRemove.bind(this, i)}></button>
            </div>
        ));
    }

    return (
        <div>
            {els}
            <button className='btn btn-sm btn-default' type='button' onClick={addEl}>Add</Button>
        </div>
        );

}

export function drawMolSearch(el, wsId, onSelected, fieldName, value) {
    ReactDOM.render((<MolSearch
            onSelected={onSelected}
            wsId={wsId}
            fieldName={fieldName}
            value={value}
        />), el);
}


export function drawMultiInput(InputType, el, onChange, initialEntries, extraAttrs) {
    ReactDOM.render((<MultiInput
        InputType={InputType}
        initialEntries={initialEntries}
        onChange={onChange}
        extraAttrs={extraAttrs}
        />), el);
}
