/**
 * Creates POST options that can be used for fetch().
 * Automatically pulls in the csrf token, JSON'ifies 'data',
 * and includes anything specified in 'extra' as FormData.
 */
function makePostOpts(data, extra) {
    const csrf = document.getElementsByName("csrfmiddlewaretoken")[0].value
    const formData = new FormData();
    formData.append('csrfmiddlewaretoken', csrf);
    if (data) {
        const queryStr = JSON.stringify(data);
        formData.append('query', queryStr);
    }
    if (extra) {
        for (const key in extra) {
            const val = extra[key];
            formData.append(key, val);
        }
    }
    const opts = {
        method: "POST",
        body: formData,
        redirect: 'manual',
    };
    return opts;
}

async function markWorkflowStatus(ws_id, status, name) {
    const query = {
        name,
        status,
    };
    const resp = await fetch(`/${ws_id}/workflow/`, makePostOpts(query, {update_btn: true}));
    const respData = await resp.json();
    const el = document.getElementById(`workflow-status-${name}`);
    el.classList = `btn btn-xs dropdown-toggle ${respData.button_classes}`;
    el.querySelector('#status').innerText = respData.status_text;
}

