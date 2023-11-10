import datetime
import matplotlib.pyplot as plt
import numpy as np

def gw_plot(train, test, 
	    	dates=None, start_date=None,  
		pred=None, model=None,
            	train_limit=None, normalized=False, 
		save_file=None):
    
    """ Plot the training data, and test data.
    
        If no `dates` or `start_date` is provided the data will be indexed by
	`Days From Start of Record`
        
        If `pred` is included, it will be assumed to be the same length as 
	`test`, and correspond to the same indices.


    Parameters
    ----------
    train : np.array
        A one dimensional array of floats.
        The output values of the training data.
    
    test : np.array
        A one dimensional array of floats.
        The output values of the test data.
    
    dates: np.array or pd.DataFrame (default: None)
        A one dimensional array of datetimes.
        These dates should correspond to the dates spanned by the training 
	data and then the test data, in order.
    
    start_date: datetime.datetime (default: None)
        This will be ignored if `dates` is not None.
        A datetime object indicating the first date of the training set data.

    pred : np.array (default: None)
        A one dimensional array of floats
        The predicted output values. This should be the same length as `test`.
    
    model : string (default: None)
        A string describing the model.
        Adds the name of the model to the title of the plot.

    train_limit : int (default: None)
        An integer indicating how many days of the training data to include in
	the plot.

    normalized : bool (default: False)
        A bool indicating whether the data has been normalized. This is used
	to determine the plot title and y-axis labels.

    save_file: string (default: None)
        A string containing a path to a filename, or a Python file-like object
	(e.g. 'my_figure.eps').
        The output format is deduced from the extension of the filename. 
    
    Returns
    -------
    None
    
    """

    len_train = train.shape[0]
    len_test = test.shape[0]

    x_label = 'Date'
    
    if train_limit==None:
        train_limit = len_train
    
    plt.figure(figsize=(12,6))

    if dates is not None:
        dt_train = dates[:len_train]
        dt_test = dates[len_train:len_train+len_test]
    elif (start_date is not None):
        dates = np.array([(start_date + datetime.timedelta(days=i)) 
                               for i in range(len_train+len_test)])
        dt_train = dates[:len_train]
        dt_test = dates[len_train:len_train+len_test]
    else:
        dates = np.arange(0,len_train+len_test,1)
        dt_train = dates[:len_train]
        dt_test = dates[len_train:len_train+len_test]
        x_label = 'Days From Start of Record'
    
    ## Plot the training data
    plt.plot(dt_train[-train_limit:], train[-train_limit:],
             	label='Train', 
             	marker='.', 
             	zorder=-10)

    ## Plot the test data
    plt.scatter(dt_test, test,
                    edgecolors='k',
                    linewidth=0.1,
                    c='#ff7f0e', 
                    alpha =1,
                    s=24,
                    label='Test')
    
    ## Plot the prediction
    if pred is not None:
        plt.scatter(dt_test, pred,
                        marker='X',
                        edgecolors='k',
                        linewidth=0.1,
                        c='red', 
                        alpha =.75,
                        s=24,
                        label='Prediction')

    if model:
        model = '\n'+model
    else:
        model = ''
    
    if normalized:
        plt.title('Water Depth from the Surface [Normalized]'+model, 
		  	fontsize = 16)
        plt.ylabel('Feet Below Surface [Normalized]', fontsize=14)
    else:    
        plt.title('Water Depth from the Surface'+model, fontsize = 16)
        plt.ylabel('Feet Below Surface', fontsize=14)

    plt.xlabel(x_label, fontsize=14)
    plt.legend(fontsize = 14)

    if save_file is not None:
        plt.savefig(save_file)

    plt.show()
