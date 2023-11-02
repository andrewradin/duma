
import * as React from 'react';
import * as ReactDOM from 'react-dom';


const kInitialFilter = 'molecule';

function PathwayFilter({filter, setFilter}) {
    const configItemStyle = {
        paddingRight: '0.5rem',
        paddingLeft: '0.5rem',
        borderLeft: '1px solid #eee',
    }

    const onFilterBy = (e) => {
        filter.filterBy = e.target.value;
        setFilter({...filter});
    };
    const onValue = (e) => {
        filter.value = e.target.value;
        setFilter({...filter});
    }
    const onDelete = () => {
        setFilter(null);
    };
    return (
            <div style={configItemStyle}>
                <a style={{'cursor':'pointer', 'float': 'right', 'color': '#a24'}} onClick={onDelete}><b>X</b></a>
                Filter By: <select style={{width: 'auto'}} className='form-control' value={filter.filterBy} onChange={onFilterBy}>
                    <option value='molecule'>Molecule Rank</option>
                    <option value='pwyscore'>Pathway Rank</option>
                    <option value='protscore'>Protein Rank</option>
                </select>
                <input className='form-control'  value={filter.value} onChange={onValue} />

            </div>
    );

}

export function PathwayFilters({filters, setFilters}) {
    const onSetFilter = (newValue, idx) => {
        if (newValue === null) {
            filters.splice(idx, 1);
        }
        setFilters([...filters]);
    };
    const addFilter = () => {
        setFilters([...filters, {filterBy: kInitialFilter}]);
    }
    const filterEls = filters.map((filter, idx) => <PathwayFilter filter={filter} setFilter={(x) => onSetFilter(x, idx)} />);

    return (
        <>
            {filterEls}
            <button className='btn btn-default' onClick={addFilter}>Add Filter</button>
        </>
    )
    
}