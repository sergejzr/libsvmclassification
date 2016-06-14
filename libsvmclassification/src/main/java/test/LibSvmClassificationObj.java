package test;



import java.io.IOException;
import java.util.Hashtable;
import java.util.Vector;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;

public class LibSvmClassificationObj{
	private svm_parameter param;		// set by parse_command_line
	private svm_problem prob;		// set by read_problem
	private svm_model model;
	/*
	private String input_file_name;		// set by parse_command_line
	private String model_file_name;		// set by parse_command_line
	private String error_msg;
	private int cross_validation;
	private int nr_fold;
	*/
	protected static svm_print_interface svm_print_null = new svm_print_interface()
	{
		public void print(String s) {}
	};
	

	
	public static void main(String[] args) throws IOException {
		LibSvmClassificationObj cmodel = new LibSvmClassificationObj();
		
		cmodel.runExperiment();
	}
	
	public void runExperiment() throws IOException
	{
		//building the train files
		Vector<Double> vy_train = new Vector<Double>();
		Vector<svm_node[]> vx_train = new Vector<svm_node[]>();			
		
	//	int count = 0;
		
		String[] postexts=new String[]{"sonne strand","sonne strand","sonne strand","sonne strand","sonne strand","sonne strand"};
		String[] negtexts=new String[]{"winter snow","winter snow","winter snow","winter snow","winter snow","winter snow"};
		String[] nutraltexts=new String[]{"winter strand","winter snow sonne","sonne strand strand sonne"};
		
		Hashtable<String, Integer> idx=new Hashtable<String, Integer>();
		
		for(String pos:postexts)
		{			
			svm_node nodeX[]=text2Node(pos,idx);
			vx_train.addElement(nodeX);				
			vy_train.addElement((double)0);						
		}
		for(String neg:negtexts)
		{			
			svm_node nodeX[]=text2Node(neg,idx);
			vx_train.addElement(nodeX);				
			vy_train.addElement((double)1);						
		}
		
	
		int max_index = idx.size()+1;
		
		//training	the model	
		setParameters();
		read_problem(vy_train, vx_train, max_index);
		System.out.println("building the model");
		model = svm.svm_train(prob,param);		
		System.out.println("done building the model");
		svm.svm_save_model("textmod.mdl", model);
		
		String[][] texts=new String[][]{postexts,negtexts,nutraltexts};
		
		for(String[] list:texts)
		for(String neg:list)
		{			
			svm_node nodeX[]=text2Node(neg,idx);
			
			double predictedValues[] = new double[2];
			double predictedProbabilities[] = new double[2];
			int[] labels = new int[2];
			
			
			svm.svm_predict_probability(model, nodeX, predictedProbabilities);
			svm.svm_predict_values(model, nodeX, predictedValues);
			svm.svm_get_labels(model, labels);	
			
			double v = svm.svm_predict(model,nodeX);
			
			System.out.println(v);	
		}
	
	}
	
	private svm_node[] text2Node(String pos, Hashtable<String, Integer> idx) {
		String[] terms=pos.split(" ");
		
		
		
		Hashtable<Integer, svm_node> feature=new Hashtable<Integer, svm_node>();
		
		for(int i=0;i<terms.length;i++)
		{
			
			
			Integer dim=null; 
			if((dim=idx.get(terms[i]))==null)
			{
				idx.put(terms[i], dim=(idx.size()+1));
			}
			 svm_node node = null;
			 if((node = feature.get(dim))==null)
			 {
				 feature.put(dim, node=new svm_node());
				 node.index=dim;
				 node.value=0;
			 }
			node.value+=1;
		}
		svm_node x[]=new svm_node[feature.size()];
		int cnt=0;
		for(Integer dim:feature.keySet())
		{
			x[cnt++]=feature.get(dim);
		}
	
	return x;
	}



	
	
	private void setParameters()
	{
	//	int i;
	//	svm_print_interface print_func = null;	// default printing to stdout

		param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.NU_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 3;
		param.gamma = 0;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 0.5;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 1;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];
	//	cross_validation = 0;
		
	}
	
	public static double atof(String s)
	{
		double d = Double.valueOf(s).doubleValue();
		if (Double.isNaN(d) || Double.isInfinite(d))
		{
			System.err.print("NaN or Infinity in input\n");
			System.exit(1);
		}
		return(d);
	}

	public static int atoi(String s)
	{
		return Integer.parseInt(s);
	}

	
	public static void exit_with_help()
	{
		System.out.print(
		 "Usage: svm_train [options] training_set_file [model_file]\n"
		+"options:\n"
		+"-s svm_type : set type of SVM (default 0)\n"
		+"	0 -- C-SVC\n"
		+"	1 -- nu-SVC\n"
		+"	2 -- one-class SVM\n"
		+"	3 -- epsilon-SVR\n"
		+"	4 -- nu-SVR\n"
		+"-t kernel_type : set type of kernel function (default 2)\n"
		+"	0 -- linear: u'*v\n"
		+"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		+"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		+"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		+"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		+"-d degree : set degree in kernel function (default 3)\n"
		+"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		+"-r coef0 : set coef0 in kernel function (default 0)\n"
		+"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		+"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		+"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		+"-m cachesize : set cache memory size in MB (default 100)\n"
		+"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		+"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		+"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		+"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		+"-v n : n-fold cross validation mode\n"
		+"-q : quiet mode (no outputs)\n"
		);
		System.exit(1);
	}

	private void read_problem(Vector<Double> vy, Vector<svm_node[]> vx , int max_index) throws IOException
	{		
	
			
		
		prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
			prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			prob.y[i] = vy.elementAt(i);

		if(param.gamma == 0 && max_index > 0)
			param.gamma = 1.0/max_index;

		if(param.kernel_type == svm_parameter.PRECOMPUTED)
			for(int i=0;i<prob.l;i++)
			{
				if (prob.x[i][0].index != 0)
				{
					System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
					System.exit(1);
				}
				if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
				{
					System.err.print("Wrong input format: sample_serial_number out of range\n");
					System.exit(1);
				}
			}
		
	}

	
	
	
}

