# Databricks notebook source
from flask import Flask, render_template, jsonify, Response,request, session, current_app,redirect, url_for
import numpy as np
import pandas as pd
import base64
import io 
import atexit
import os
import seaborn as sns
from IPython import get_ipython
import base64
import matplotlib.pyplot as plt
import random

from werkzeug.serving import WSGIRequestHandler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from qiskit.primitives import Sampler
from qiskit import Aer
from qiskit import transpile, execute
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import SPSA,COBYLA
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider
from qiskit.visualization import circuit_drawer
from qiskit.visualization import plot_histogram

import time


app = Flask(__name__)

class Calculations:
    def __init__(self):
        self.kernel_process = False
        self.vector_process = False
        self.var_circuit = None
        self.training_complete = False
        self.progress = 0
        
    def kernel_jobs(self, spinner):
        if spinner == "success":
            self.kernel_process = True
        elif spinner == "completed":
            self.vector_process = True
                
    def qsettings(self, var_circuit):
        # ... your training code ...
        self.var_circuit = var_circuit
        self.training_complete = True
        self.progress = 0
        
    def run_quantum_job(self):
        if self.var_circuit is not None:
            provider = AzureQuantumProvider(
                resource_id="/subscriptions/e7eef170-5f0d-42c3-a6c6-9ae33096de85/resourceGroups/AzureQuantum/providers/Microsoft.Quantum/Workspaces/Quantamunia",
                location="West Europe"
            )
            provider.backends()
            backend = provider.get_backend("ionq.simulator")
            params = self.var_circuit.parameters
            random_values = [random.uniform(0, 2 * np.pi) for _ in range(len(params))]
            bound_circuit = self.var_circuit.bind_parameters(dict(zip(params, random_values)))
            total_iterations = 10
            progress_values = []
            job = execute(bound_circuit, backend=backend)
            result = job.result()
            counts = result.get_counts()     
           
            fig = plot_histogram(counts)
            fig.text(0.5, 0.95, 'Circuit Histogram', ha='center', fontsize=12)
            fig.savefig('static/histogram.jpg')
            plt.close(fig)
            
              # Calculate the total number of measurements
            total_measurements = sum(counts.values())
            # Calculate the probabilities of each state
            probabilities = {state: count / total_measurements for state, count in counts.items()}
            # Calculate the Shannon entropy of the probability distribution
            entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
            
            # Identify the most probable outcome (state with highest probability)
            most_probable_state = max(probabilities, key=probabilities.get)
            metrics = {
            'TotalMeasurements': total_measurements,
            'Probabilities': probabilities,
            'Entropy': entropy,
            'MostProbableState': most_probable_state
            }
            return metrics  # or return a result if needed
        
        
        
        
        
            
                
            
            

spinObj = Calculations()
azureJob = Calculations()

objective_func_vals = []


@app.route('/',methods=['GET', 'POST'])
def index():
    
 
   
    if request.method == 'POST':
        
        model = request.form['model']
        dataset = request.form['dataset']
        feature_map_type = request.form['featuremap']
        optimizer_type = request.form['optimizer']
        split = request.form['split']
        pca_no = request.form['pca']
        pca_no = int(pca_no)
        entang = request.form['entang']
       
        
        
        
        X,y,dset= load_dataset(dataset)
        
        feature_no = X.shape[1]
        
        X = pca(pca_no, X)
        
        test_size = split_ratio(split)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
        
        
       
    
        feature_map, ansatz, base64_feature_map_image, base64_ansatz_image = create_feature_map(feature_map_type, pca_no, entang)
        
        optimizer, optname = create_optimizer(optimizer_type)    
        
        import threading
        def train(model,optimizer,feature_map,ansatz,X_train_scaled, X_test_scaled, y_train, y_test,dset):
            
                load_model(model,optimizer,feature_map,ansatz,X_train_scaled, X_test_scaled, y_train, y_test,dset)
            
            
            
        train_thread = threading.Thread(target=train, args=(model,optimizer,feature_map,ansatz,X_train_scaled, X_test_scaled, y_train, y_test,dset))
        train_thread.start()
        
        return render_template('index.html',fmap=base64_feature_map_image, ans = base64_ansatz_image, dname = dataset, pcano=pca_no,
                               ent = entang, opt = optname, mod = model, f_no = feature_no)
    
    
       
          
       
    #return redirect(url_for('index'))
    return render_template('index.html')


    
def load_dataset(dataset):
    if dataset == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        return X, y, iris
    if dataset == 'diabetes':
        diabetes = datasets.load_diabetes()
        X = diabetes.data
        y = diabetes.target
        return X, y,diabetes
    if dataset == 'wine':
        wine = datasets.load_wine()
        X = wine.data
        y = wine.target
        return X, y,wine
    if dataset == 'cancer':
        cancer = datasets.load_breast_cancer()
        X = cancer.data
        y = cancer.target
        return X, y,cancer
    
def split_ratio(split):
    if split == '80/20':
        return 0.2
    else :
        return 0.3

def preprocess_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def draw_and_encode_circuit(circuit):
    image_stream = io.BytesIO()
    circuit_drawer(circuit, output="mpl", fold=20, filename=image_stream)
    image_stream.seek(0)
    base64_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    return base64_image

def create_feature_map(feature_map_type, fdimension, entang):
    if feature_map_type == 'zz':
        # Create the ZZFeatureMap and draw its circuit
        feature_map = ZZFeatureMap(feature_dimension=fdimension, reps=1, entanglement=entang)
        base64_feature_map_image = draw_and_encode_circuit(feature_map.decompose())

        # Create the RealAmplitudes ansatz and draw its circuit
        ansatz = RealAmplitudes(num_qubits=fdimension, reps=3)
        base64_ansatz_image = draw_and_encode_circuit(ansatz.decompose())

        return feature_map,ansatz, base64_feature_map_image, base64_ansatz_image
    
    elif feature_map_type == 'z':
        # Create the ZFeatureMap and draw its circuit
        feature_map = ZFeatureMap(feature_dimension=fdimension, reps=1)
        base64_feature_map_image = draw_and_encode_circuit(feature_map.decompose())

        # Create the RealAmplitudes ansatz and draw its circuit
        ansatz = RealAmplitudes(num_qubits=fdimension, reps=3)
        base64_ansatz_image = draw_and_encode_circuit(ansatz.decompose())

        return feature_map,ansatz,base64_feature_map_image, base64_ansatz_image
    
    elif feature_map_type == 'pauli':
        # Create the ZZFeatureMap and draw its circuit
        feature_map = PauliFeatureMap(feature_dimension=fdimension, reps=1, entanglement=entang)
        base64_feature_map_image = draw_and_encode_circuit(feature_map.decompose())

        # Create the RealAmplitudes ansatz and draw its circuit
        ansatz = RealAmplitudes(num_qubits=fdimension, reps=3)
        base64_ansatz_image = draw_and_encode_circuit(ansatz.decompose())

        return feature_map,ansatz,base64_feature_map_image, base64_ansatz_image

def create_optimizer(optimizer_type):
    if optimizer_type == 'spsa':
        return SPSA(maxiter=40), "SPSA"
    elif optimizer_type == 'cobyla':
        return COBYLA(maxiter=40), "COBYLA"
    elif optimizer_type == 'adam':
        return COBYLA(maxiter=40), "ADAM"
    else:
        return "NONE", "NONE"
    
def pca(pca_no,X):
    pca = decomposition.PCA(n_components=pca_no)
    pca.fit(X)
    X = pca.transform(X)
    return X   
    
    
     
#model,optimizer,feature_map,ansatz, X_train_scaled, X_test_scaled, y_train, y_test

def load_model(model,optimizer,feature_map,ansatz,X_train_scaled, X_test_scaled, y_train, y_test,dset):
   
    if model == "VQC/MLP":
        c_metrics = mlp_classifier(X_train_scaled, X_test_scaled, y_train, y_test,dset)
        q_metrics = vqc_classifier(optimizer,feature_map,ansatz, X_train_scaled, X_test_scaled, y_train, y_test,dset)        
        barchart_mlp_vqc(model,c_metrics,q_metrics)
        
    
    if model == "QSVM/SVM":
        c_metrics = svm_classifier(X_train_scaled, X_test_scaled, y_train, y_test,dset)
        q_metrics = qsvm_classifier(feature_map,X_train_scaled, X_test_scaled, y_train, y_test,dset)
        barchart_mlp_vqc(model,c_metrics,q_metrics)
        
    
    if model == "QSVC/SVC":
        c_metrics = svc_classifier(X_train_scaled, X_test_scaled, y_train, y_test,dset)
        q_metrics = qsvc_classifier(feature_map,X_train_scaled, X_test_scaled, y_train, y_test,dset)
        barchart_mlp_vqc(model,c_metrics,q_metrics)
    
    if model == "VQC/LR":
        c_metrics = lr_classifier(X_train_scaled, X_test_scaled, y_train, y_test,dset)
        q_metrics = vqc_classifier(optimizer,feature_map,ansatz,X_train_scaled, X_test_scaled, y_train, y_test,dset)
        barchart_mlp_vqc(model,c_metrics,q_metrics)

#=============================== MLP / VQC CLASSIFIER==========================


def mlp_classifier(X_train_scaled, X_test_scaled, y_train, y_test,dset):
    
    # Create an instance of the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=550, random_state=42)
    # Train the MLP classifier
    mlp.fit(X_train_scaled, y_train)
    # Make predictions with MLP classifier
    y_pred_mlp = mlp.predict(X_test_scaled)
    # Calculate accuracy, F1 score, precision, and recall for MLP classifier
    cm = confusion_matrix(y_test, y_pred_mlp)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')
    precision_mlp = precision_score(y_test, y_pred_mlp, average='weighted')
    recall_mlp = recall_score(y_test, y_pred_mlp, average='weighted') 
    class_name = "MLP"
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {class_name}', classes=dset.target_names, filename='static/cc_matrix.jpg')
    print(accuracy_mlp)
    print(precision_mlp)
    print(recall_mlp)
    print(f1_mlp)
    return accuracy_mlp, f1_mlp, precision_mlp, recall_mlp





def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration", fontweight="bold", fontsize=13, y=1.05)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Objective function value",fontsize=12)
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.savefig('static/graph.jpg')  # Save the graph as an image
    plt.close()
    
    
def vqc_classifier(optimizer,feature_map,ansatz, X_train_scaled, X_test_scaled, y_train, y_test,dset):
    
    
    sampler = Sampler()
    vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,

    )

    #import threading
    #def run_callback():
    vqc.fit(X_train_scaled, y_train)

    #train_thread = threading.Thread(target=run_callback)
    #train_thread.start()
    
    # Wait for the training thread to finish
    #train_thread.join()


    # Make predictions with VQC
    y_pred_vqc = vqc.predict(X_test_scaled)
    # Calculate accuracy, F1 score, precision, and recall for VQC
    cm = confusion_matrix(y_test, y_pred_vqc)
    accuracy_vqc = accuracy_score(y_test, y_pred_vqc)
    f1_vqc = f1_score(y_test, y_pred_vqc, average='weighted')
    precision_vqc = precision_score(y_test, y_pred_vqc, average='weighted')
    recall_vqc = recall_score(y_test, y_pred_vqc, average='weighted')
    print(accuracy_vqc)
    print(f1_vqc)
    print(precision_vqc)
    print( recall_vqc)
    spinner = "completed"
    spinObj.kernel_jobs(spinner)
    class_name = "VQC"
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {class_name}', classes=dset.target_names, filename='static/qc_matrix.jpg')
    azureJob.qsettings(vqc.circuit)
    return accuracy_vqc ,f1_vqc,precision_vqc, recall_vqc

def barchart_mlp_vqc(model, c_metrics, q_metrics):
    metrics = {}
    quantum, classical = model.split('/')
    # Bar chart
    metrics['c'] = c_metrics
    metrics['q'] = q_metrics
    accuracy_c, f1_c, precision_c, recall_c = metrics['c']
    accuracy_q, f1_q, precision_q, recall_q = metrics['q']
    accuracy_c = round(accuracy_c,2)
    f1_c=round(f1_c,2)
    precision_c=round(precision_c,2)
    recall_c=round(recall_c,2)
    accuracy_q = round(accuracy_q,2)
    f1_q=round(f1_q,2)
    precision_q=round(precision_q,2)
    recall_q=round(recall_q,2)
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    c_scores = [accuracy_c, precision_c, recall_c, f1_c]
    q_scores = [accuracy_q, precision_q, recall_q, f1_q]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, c_scores, width, label=classical)
    rects2 = ax.bar(x + width/2, q_scores, width, label=quantum)
    ax.margins(y=0.15)
    ax.set_ylabel('Scores')
    ax.set_title(f'Comparison of Metrics: {classical} vs {quantum}', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1, 1))

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # Save the plot as a JPG image in the static folder
    plt.savefig('static/barchart.jpg')

    # Close the plot to free up resources
    plt.close()
 


 



#================================= QSVM/SVM======================================================

def svm_classifier(X_train_scaled, X_test_scaled, y_train, y_test,dset):
    
    svm = SVC(kernel='linear')
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    
    
  
    cm = confusion_matrix(y_test, svm_pred)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_precision = precision_score(y_test, svm_pred, average='macro')
    svm_recall = recall_score(y_test, svm_pred, average='macro')
    svm_f1 = f1_score(y_test, svm_pred, average='macro')
    print(svm_accuracy)
    print(svm_f1)
    print(svm_precision)
    print( svm_recall)
    class_name = "SVM"
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {class_name}', classes=dset.target_names, filename="static/cc_matrix.jpg")
    return svm_accuracy,svm_precision, svm_recall,svm_f1
    
    
def qsvm_classifier(feature_map, X_train_scaled, X_test_scaled, y_train, y_test,dset):
    
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=Aer.get_backend('qasm_simulator'))
    qsvm = QSVC(quantum_kernel=qkernel)
    #start = time.time()
    #epochs = 4
    #for _ in tqdm(range(epochs), desc="Training Progress"):
    qsvm.fit(X_train_scaled, y_train)
    #elapsed = time.time() - start
    # Predict labels for the test set
    y_pred = qsvm.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
   
    qsvm_accuracy = np.sum(y_pred == y_test) / len(y_test)
    qsvm_f1 = f1_score(y_test, y_pred, average='weighted')
    qsvm_precision = precision_score(y_test, y_pred, average='weighted')
    qsvm_recall = recall_score(y_test, y_pred, average='weighted')
    print(qsvm_accuracy)
    print(qsvm_f1)
    print(qsvm_precision)
    print( qsvm_recall)
    spinner = "success"
    spinObj.kernel_jobs(spinner)
    class_name = "QSVM"
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {class_name}', classes=dset.target_names, filename="static/qc_matrix.jpg")
    return qsvm_accuracy,qsvm_f1,qsvm_precision,qsvm_recall
  

#================================QSVC/SVC======================================================

def svc_classifier(X_train_scaled, X_test_scaled, y_train, y_test,dset):
    svc = SVC()
    svc.fit(X_train_scaled, y_train) 
    svc_pred = svc.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, svc_pred)
    svc_accuracy = accuracy_score(y_test, svc_pred)
    svc_precision = precision_score(y_test, svc_pred, average='macro')
    svc_recall = recall_score(y_test, svc_pred, average='macro')
    svc_f1 = f1_score(y_test, svc_pred, average='macro')
    class_name = "SVC"
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {class_name}', classes=dset.target_names, filename="static/cc_matrix.jpg")
    return svc_accuracy,svc_precision, svc_recall,svc_f1

def qsvc_classifier(feature_map, X_train_scaled, X_test_scaled, y_train, y_test,dset):
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)
    kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=kernel)
    #start = time.time()
    qsvc.fit(X_train_scaled, y_train)
    #elapsed = time.time() - start
    y_pred = qsvc.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    qsvc_accuracy = np.sum(y_pred == y_test) / len(y_test)
    qsvc_f1 = f1_score(y_test, y_pred, average='weighted')
    qsvc_precision = precision_score(y_test, y_pred, average='weighted')
    qsvc_recall = recall_score(y_test, y_pred, average='weighted')  
    spinner = "success"
    spinObj.kernel_jobs(spinner)
    class_name = "QSVC"
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {class_name}', classes=dset.target_names, filename="static/qc_matrix.jpg")
    return qsvc_accuracy,qsvc_f1,qsvc_precision,qsvc_recall


#================================VQC/LR======================================================


    
def lr_classifier(X_train_scaled, X_test_scaled, y_train, y_test,dset):
    # Create a logistic regression CL_model_LR
    lr = LogisticRegression()
    # Train the CL_model_LR
    lr.fit(X_train_scaled,  y_train)
    # Make predictions on the test set
    y_pred = lr.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    lr_accuracy = accuracy_score(y_test, y_pred)
    lr_precision = precision_score(y_test, y_pred, average='macro')
    lr_recall = recall_score(y_test, y_pred, average='macro')
    lr_f1 = f1_score(y_test, y_pred, average='macro')
    class_name = "LR"
    plot_confusion_matrix(cm, title=f'Confusion Matrix for {class_name}', classes=dset.target_names, filename="static/cc_matrix.jpg")
    return lr_accuracy,lr_precision, lr_recall,lr_f1



# Create a function to display the confusion matrix in a smaller size
def plot_confusion_matrix(cm, title, classes, filename):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


    


def ansatz_cir(N):
    image_stream = io.BytesIO()
    ansatz = RealAmplitudes(num_qubits=N, reps=3)
    ansatz.decompose().draw(output="mpl", fold=20,filename=image_stream )
    image_stream.seek(0)
    base64_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    return base64_image

    
@app.route('/get_graph_data')
def get_graph_data():
    # Return the current timestamp as a unique parameter to prevent caching
    timestamp = int(time.time() * 1000)
    return jsonify({'graph_src': f'static/graph.jpg?{timestamp}'})

@app.route('/get_barchart')
def get_barchart():
    # Return the current timestamp as a unique parameter to prevent caching
    timestamp = int(time.time() * 1500)
    return jsonify({'bar_src': f'static/barchart.jpg?{timestamp}'})

@app.route('/get_cc_matrix')
def get_cc_matrix():
    # Return the current timestamp as a unique parameter to prevent caching
    timestamp = int(time.time() * 2000)
    return jsonify({'bar_src': f'static/cc_matrix.jpg?{timestamp}'})

@app.route('/get_qc_matrix')
def get_qc_matrix():
    # Return the current timestamp as a unique parameter to prevent caching
    timestamp = int(time.time() * 2500)
    return jsonify({'bar_src': f'static/qc_matrix.jpg?{timestamp}'})

@app.route('/quantum_job')
def quantum_job_route():
    if azureJob.training_complete:
        metrics = azureJob.run_quantum_job()
        azureJob.training_complete = False
        
        return jsonify({'result': 'success', 'progress': 0, 'metrics': metrics})
    else:
        return jsonify({'result': 'training_in_progress', 'progress': azureJob.progress})


@app.route('/get_hist')
def get_hist():
     # Return the current timestamp as a unique parameter to prevent caching
    timestamp = int(time.time() * 4000)
    return jsonify({'hist_src': f'static/histogram.jpg?{timestamp}'})

@app.route('/q_spinner')
def q_spinner():
    if spinObj.vector_process:
        qvector =  "completed"
        spinObj.vector_process = False
        return jsonify({"spinner": qvector})
    
    if spinObj.kernel_process:
        spinner = "success"
        spinObj.kernel_process = False
        return jsonify({"spinner": spinner})
    else:
        spinner = "training"
        return jsonify({"spinner": spinner})
    
       

      



def delete_png_and_jpg_images():
    static_folder = os.path.join(os.getcwd(), 'static')
    extensions_to_delete = ['.jpg', '.png']
    for filename in os.listdir(static_folder):
        if any(filename.lower().endswith(ext) for ext in extensions_to_delete):
            file_path = os.path.join(static_folder, filename)
            os.remove(file_path)
  

if __name__ == '__main__':
   
    atexit.register(delete_png_and_jpg_images)
    app.run(port=5889, debug=False, use_reloader=False)   

