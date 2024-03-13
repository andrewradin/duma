from django.shortcuts import render
from django.contrib.auth.decorators import login_required

import datetime

from path_helper import PathHelper
from runner.models import Process

# Create your views here.
@login_required
def jobs(request):
    return render(request
                ,'runner/jobs.html'
                ,{'ph':PathHelper
                 ,'qs':Process.active_jobs_qs().order_by('-id')
                 ,'repeat':10
                 ,'now':datetime.datetime.now()
                 ,'heading':'All Active Background Jobs'
                 }
                )
