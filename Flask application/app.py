import pickle
import mne
from flask import Flask,request,app,url_for,jsonify,render_template
import numpy as np
import pandas as pd
import sklearn
# from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
# scaler = StandardScaler(n_features=9)
def read_data(file_path):
  content=mne.io.read_raw_eeglab(file_path, eog=(), preload=True)
  content = content.drop_channels(["T","X","Y"])
  return content
def mean(x):
  return np.mean(x)
def median(x):
  return np.median(x)
def std(x):
  return np.std(x)
def var(x):
  return np.var(x) 
def shannon(x):
 y=np.power(x,2)  
 y1=np.sum(y)
 pe=y/y1
 se=np.sum(pe*np.log(np.power(pe,2)))
 return se
def logenergy(x):
  y=np.power(x,2)
  y1=np.sum(y)
  pe=y/y1
  lee=np.sum(pe*np.log(pe))
  return lee
def kurtosis(a):
    b = a # Extracting the data from the 14 channels
    output = np.zeros(len(b)) # Initializing the output array with zeros (length = 14)
    k = 0; # For counting the current row no.
    for i in b:
        mean_i = np.mean(i) # Saving the mean of array i
        std_i = np.std(i) # Saving the standard deviation of array i
        t = 0.0
        for j in i:
            t += (pow((j-mean_i)/std_i,4)-3)
        kurtosis_i = t/len(i) # Formula: (1/N)*(summation(x_i-mean)/standard_deviation)^4-3
        output[k] = kurtosis_i # Saving the kurtosis in the array created
        k +=1 # Updating the current row no.
    return np.sum(output)
def compute_hjorth_mobility(data):
    """Hjorth mobility (per channel).
    Hjorth mobility parameter computed in the time domain.
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **hjorth_mobility**. See [1]_.
    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """
    x = np.insert(data, 0, 0)
    dx = np.diff(x)
    sx = np.std(x, ddof=1)
    sdx = np.std(dx, ddof=1)
    mobility = np.divide(sdx, sx)
    return mobility
def compute_hjorth_complexity(data):
    """Hjorth complexity (per channel).
    Hjorth complexity parameter computed in the time domain.
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **hjorth_complexity**. See [1]_.
    References
    ----------
    .. [1] Paivinen, N. et al. (2005). Epileptic seizure detection: A
           nonlinear viewpoint. Computer methods and programs in biomedicine,
           79(2), 151-159.
    """
    x = np.insert(data, 0, 0)
    dx = np.diff(x)
    m_dx = compute_hjorth_mobility(dx)
    m_x = compute_hjorth_mobility(data)
    complexity = np.divide(m_dx, m_x)
    return complexity
@app.route("/")
def home():
    return render_template("home.html")
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1,-1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     output=model.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data1=[float(x) for x in request.form.values()]
    # filedata=form['myfile']
    # set_contents = request.files('myfile')
    
    file = request.files['myfile']
    file_path = file.filename
    print("file",file_path)

    data=read_data("/home/rucha/Desktop/Fatigue-main/setfiles/"+ file_path)
    
    # data=read_data("/home/rucha/Desktop/Fatigue-main/setfiles/subject_20.set")
    # print(set_contents)
    # print("set file path", set_file)
    print("data",data.info)
    
    print("hello")
    print("data1",data1)
    print(type(data1))
    # print("mean ",mean(data.get_data()))
    # print("shannon ",shannon(data.get_data()))
    # print("logenergy ",logenergy(data.get_data()))
    # print("kurtosis ",kurtosis(data.get_data()))
    # print("compute_hjorth_mobility ",compute_hjorth_mobility(data.get_data()))
    # print("compute_hjorth_complexity ",compute_hjorth_complexity(data.get_data()))
    # mean1=mean(data.get_data())
    # print("mean",mean1)
    mean1=mean(data.get_data())
    median1=median(data.get_data())
    std1=std(data.get_data())
    var1=var(data.get_data())
    shannon1=shannon(data.get_data())
    logenergy1=logenergy(data.get_data())
    kurtosis1=kurtosis(data.get_data())
    compute_hjorth_mobility1=compute_hjorth_mobility(data.get_data())
    compute_hjorth_complexity1=compute_hjorth_complexity(data.get_data())
    features=[]
    features.append(mean1)
    # features.append(median1)
    # features.append(var1)
    # features.append(std1)
    features.append(shannon1)
    features.append(logenergy1)
    features.append(kurtosis1)
    features.append(compute_hjorth_mobility1)
    features.append(compute_hjorth_complexity1)
   
    
    print("features",features)
    print(type(features))
    # new_features = features[0]
    # print("new features",new_features)



    final_input=scalar.transform(np.array(features).reshape(1,-1))
    print("Final Input",final_input)
    output=model.predict(final_input)[0]
    print("output",output)
    return render_template("home.html",prediction_text="Subject status: {}".format(output))
    # return set_contents

    
if __name__=="__main__":
    app.run(debug=True)