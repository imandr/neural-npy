from threading import Thread, RLock
from wsgiref.simple_server import make_server
from wsgiref.util import request_uri
import json, os
import numpy as np

def synchronized(fcn):
    def f(self, *params, **args):
        with self.Lock:
            return fcn(self, *params, **args)
    return f
    
def defultEnhancer(x, y):
    # assumes ys are [0...1]
    jmax = np.argmax(y, axis=1)
    y1 = np.zeros_like(y)
    for k in range(len(y)):
        y1[k, jmax[k]] = 1.0
    return y1
        
class UnsupervisedTrainingThread(Thread):
    
    def __init__(self, mgr, model, tdata, vdata, enhancer, epochs, eta, 
                    eta_decay, report_samples,
                    mb_size, validate_size, 
                    validate_samples,   # run validation after so many samples. 
                                        # None means validate at the end of epoch
                    randomize, epoch_limit, normalize_grads):
                    
        Thread.__init__(self)
                    
        self.Model = model
        self.TData = tdata      # generator
        self.VData = vdata      # data, (X,Y)
        self.Epochs = epochs
        self.Eta = eta
        self.EtaDecay = eta_decay
        self.ReportSamples = report_samples
        self.MbSize = mb_size
        self.Randomize = randomize
        self.EpochLimit = epoch_limit
        self.ValidateSize = validate_size or len(self.VData[0])
        self.ValidateSamples = validate_samples
        self.Manager = mgr
        self.NormalizeGrads = normalize_grads
        self.Lock = RLock()
        self.Stop = False
        self.Enhancer = enhancer

    def validate(self, nsamples, tloss, terror):
        pass
        
    def run(self):
        
        nsamples = 0
        next_report = self.ReportSamples
        next_validate = self.ValidateSamples
        eta = self.Eta
        for epoch in range(self.Epochs):
            nepochsamples = 0
            #print "epoch: ", epoch, "/", self.Epochs
            for bx, by in self.TData.batches(self.MbSize, randomize=self.Randomize, max_samples = self.EpochLimit):
                #print "train: bx:", bx.shape
                y = self.Model(bx)
                y_ = self.Enhancer(bx, y)   
                #print "enhancer: y:", y, "    y_:", y_             
                y, tloss, terror = self.Model.train(bx, y_, eta, normalize_grads = self.NormalizeGrads)
                #print nepochsamples, tloss, terror
                nepochsamples += len(by)
                nsamples += len(by)
                #print "samples:", nsamples, " next report:", next_report
                if next_report != None and nsamples >= next_report:
                    self.Manager.addHistory(nsamples, tloss, terror, None, None)
                    self.Manager.reportCallback(nsamples, epoch, nepochsamples, bx, by, y, tloss, terror)
                    next_report += self.ReportSamples
                if next_validate != None and nsamples >= next_validate:
                    self.validate(nsamples, tloss, terror)
                    next_validate += self.ValidateSamples
                if self.EpochLimit and nepochsamples >= self.EpochLimit:
                    break
                    
                self.Manager.endOfBatchCallback(nsamples, epoch, nepochsamples, bx, by, y, tloss, terror)
                if self.Stop:   return
            
            if self.ValidateSamples is None:
                self.validate(nsamples, tloss, terror)
            self.Manager.endOfEpochCallback(nsamples, epoch, nepochsamples, bx, by, y, tloss, terror)     
            if self.Stop:   break
        print "Training complete"
            
    def stop(self):
        self.Stop = True
                
HomeHTML = open(os.path.join(os.path.dirname(__file__), "trainer_home_xy.html"), "r").read()

class ReportThread(Thread):
    
    HomeHTMLFile = os.path.join(os.path.dirname(__file__), "trainer_home_xy.html")
    
    def __init__(self, mgr, port):
        Thread.__init__(self)
        self.Manager = mgr
        self.Port = port
        self.Server = make_server('127.0.0.1', self.Port, self.app)
        self.Stop = False
    
    def app(self, environ, start_response):
        uri = environ["PATH_INFO"]
        if uri == "/home":
            return self.home(environ, start_response)
        elif uri == "/data":
            return self.data_json(environ, start_response)
        else:
            start_response("400 Not found", [('Content-type','text/plain')])
            out = ["%s = %s\n" % (k, v) for k, v in environ.items()]
            return ["URL %s not found\n" % (uri,)] + out
            
    def makeJSON(self, data):
        lst = [
            {   "samples":s, 
                "tloss":tl, 
                "terror":te if te else None, 
                "vloss":vl, 
                "verror":ve  if ve else None
            } 
                for s, tl, te, vl, ve in data
        ]
        return json.dumps(lst)
        
    def smooth(self, data):
        if not data:    return data
        nrows = len(data)
        ncols = len(data[0])-1
        data = np.array(data)
        data1 = data.copy()
        
        for c in range(ncols):
            vmax = None
            vmin = None
            for r in range(nrows):
                x = data[r, c+1]
                if not x is None:
                    if vmax is None:  
                        vmax = x
                        vmin = x
                    else:
                        if x > vmax:    
                            vmax = 0.9*x + 0.1*vmax
                        else:
                            vmax = 0.1*x + 0.9*vmax
                        if x < vmin:
                            vmin = 0.9*x + 0.1*vmin
                        else:
                            vmin = 0.1*x + 0.9*vmin
                    ma = (vmin + vmax)/2
                    data1[r, c+1] = ma
            for r in range(nrows):
                print data[r,c+1], data1[r, c+1]
        return data1
            
    def json_data(self):
        data = self.Manager.getHistory()
        #data = self.smooth(data)
        #print "data:", data
        return self.makeJSON(data)

    def home(self, environ, start_response):
        start_response('200 OK', [('Content-type','text/html')])
        template = open(self.HomeHTMLFile, "r").read()
        return [template % {"json_data":self.json_data()}]
        
    def data(self, environ, start_response):
        data = self.Manager.getHistory()
        data = self.makeJSON(data)
        start_response('200 OK', [('Content-type','text/json')])
        return [data]
        
    def stop(self):
        self.Stop = True
        
    def run(self):
        while not self.Stop:
            self.Server.handle_request()

class TrainerCallbackDelegate:
    
    def endOfEpochCallback(self, *params, **args):
        pass

    def endOfBatchCallback(self, *params, **args):
        pass

    def reportCallback(self, *params, **args):
        pass

    def validateCallback(self, *params, **args):
        pass

        
class UnsupervisedTrainer:
    
    def __init__(self, train_generator, validate_data, model, port, 
            enhancer=defultEnhancer, callback_delegate=None):
        
        self.RData = train_generator
        self.VData = validate_data
        self.Model = model
        self.History = []       # [(samples, tloss, terror, vloss, verror)]
        self.Lock = RLock()
        self.Port = port
        self.CallbackDelegate = callback_delegate
        self.Enhancer = enhancer
    
    @synchronized    
    def endOfEpochCallback(self, nsamples, epoch, nepochsamples, x, y, y_, loss, error):
        if self.CallbackDelegate != None:
                self.CallbackDelegate.endOfEpochCallback(self, self.Model, nsamples, epoch, nepochsamples, x, y, y_, loss, error)
        
    @synchronized    
    def reportCallback(self, nsamples, epoch, nepochsamples, x, y, y_, loss, error):
        if self.CallbackDelegate != None:
                self.CallbackDelegate.reportCallback(self, self.Model, nsamples, epoch, nepochsamples, x, y, y_, loss, error)
        
    @synchronized    
    def endOfBatchCallback(self, nsamples, epoch, nepochsamples, x, y, y_, loss, error):
        if self.CallbackDelegate != None:
                self.CallbackDelegate.endOfBatchCallback(self, self.Model, nsamples, epoch, nepochsamples, x, y, y_, loss, error)

    @synchronized    
    def validateCallback(self, nsamples, epoch, nepochsamples, x, y, y_, loss, error):
        #print "validate callback: epoch=", epoch
        if self.CallbackDelegate != None:
                self.CallbackDelegate.validateCallback(self, self.Model, nsamples, epoch, nepochsamples, x, y, y_, loss, error)
        
    @synchronized    
    def addHistory(self, nsamples, tloss, terror, vloss, verror):
            s = None if not self.History else self.History[-1][0]
            if s == nsamples:
                # merge into the last mesurement
                vals = list(self.History[-1])
                new_vals = (s, tloss, terror, vloss, verror)
                vals = [v if nv is None else nv for v, nv in zip(vals, new_vals)]
                self.History[-1] = vals
            else:
                self.History.append((nsamples, tloss, terror, vloss, verror))
            
    @synchronized    
    def getHistory(self):
            return self.History[:]
            
    def stopTraining(self):
        self.TThread.stop()
        
            
    def startTraining(self, epochs, eta, 
                    eta_decay = 0.0, report_samples = None,
                    train_mb_size = 100, validate_size = None, validate_samples = None,
                    randomize = True, epoch_limit = None, normalize_grads = False):
                    
        self.TThread = UnsupervisedTrainingThread(self, self.Model, self.RData, self.VData,
                self.Enhancer, 
                epochs, eta, eta_decay, report_samples,
                train_mb_size, validate_size, validate_samples, randomize, epoch_limit,
                normalize_grads)
                
        self.RThread = ReportThread(self, self.Port)
        
        self.TThread.start()
        self.RThread.start()
        
    def wait(self):
        self.TThread.join()
        self.RThread.join()
        
    
        

ReportThread.HomeHTML = open(os.path.join(os.path.dirname(__file__), "trainer_home_xy.html"), "r").read()



"""
<html>

<head>
    <meta http-equiv="refresh" content="10">
    <script src="https://www.amcharts.com/lib/3/amcharts.js"></script>
    <script src="https://www.amcharts.com/lib/3/xy.js"></script>
    <script src="https://www.amcharts.com/lib/3/serial.js"></script>
</head>

<body>

<table>
    <tr>
        <td><div id="loss_chart" style="width: 600px; height: 400px; border:1px solid gray"></div></td>
        <td><div id="error_chart" style="width: 600px; height: 400px; border:1px solid gray"></div></td>
    </td>
</table>
    
<script type="text/javascript">
    var data = %(json_data)s;      // data goes here
    
    function copy(x)
    {
        var o = {};
        for( k in x )
            o[k] = x[k];
        return o;
    }
    
    var opts_loss = {
        "type": "serial", 
        "dataProvider": data,
        "categoryField": "samples",
        "titles": [
            {   "text":"Loss"   }
        ], 
        "legend": {
            "align": "center"
        },
        "graphs": [
            {
                "balloonText": "samples:[[category]]<br/>training loss:[[value]]", 
                "title": "Training loss", 
                "bullet":  "round",
                "bulletSize":  3,
                "lineAlpha": 1.0, 
                "valueField":  "tloss"
            },
            {
                "balloonText": "samples:[[category]]<br/>validation loss:[[value]]", 
                "lineThickness": 2,
                "lineColor": "#223355",
                "title": "Validation loss", 
                "bullet":  "round",
                "bulletSize":  3,
                "lineAlpha": 1.0, 
                "valueField":  "vloss"
            }
        ]
    };

    var opts_errors = {
        "type": "serial", 
        "dataProvider": data,
        "categoryField": "samples",
        "titles": [
            {   "text":"Error"   }
        ], 
        "legend": {
            "align": "center"
        },
    	//"valueAxes": [
    	//	{
    	//		"id": "ValueAxis-1",
    	//		"logarithmic": true
    	//	}
        //],
        "graphs": [
            {
                "balloonText": "samples:[[category]]<br/>training error:[[value]]", 
                "title": "Training error", 
                "bullet":  "round",
                "bulletSize":  3,
                "lineAlpha": 1.0, 
                "valueField":  "terror"
            },
            {
                "balloonText": "samples:[[category]]<br/>validation error:[[value]]", 
                "lineThickness": 2,
                "lineColor": "#223355",
                "title": "Validation error", 
                "bullet":  "round",
                "bulletSize":  3,
                "lineAlpha": 1.0, 
                "valueField":  "verror"
            }
        ]
    };


    var loss_chart = AmCharts.makeChart("loss_chart", opts_loss );
    var error_chart = AmCharts.makeChart("error_chart", opts_errors );

</script>
</body>

</html>
"""

            
                
            