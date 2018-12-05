package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Hashtable;
import java.util.Vector;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;

public class LibSvmClassifier {
	private svm_parameter param; // set by parse_command_line
	private svm_problem prob; // set by read_problem
	private svm_model model;
	private Options options;
	private String indir;
	private File inputdir;
	private File svmmodelfile;
	private String configuration;
	private File inputdirtest;
	/*
	 * private String input_file_name; // set by parse_command_line private
	 * String model_file_name; // set by parse_command_line private String
	 * error_msg; private int cross_validation; private int nr_fold;
	 */
	protected static svm_print_interface svm_print_null = new svm_print_interface() {
		public void print(String s) {
		}
	};

	public LibSvmClassifier(String args[]) {

		// create Options object
		options = new Options();
		// add t option

		options.addOption(Option.builder().longOpt("inputdirtrain").hasArg(true)
				.desc("Input directory to construct the flower from").required(false).build());

		options.addOption(Option.builder().longOpt("inputdirtest").hasArg(true)
				.desc("Input directory to construct the flower from").required(false).build());

		options.addOption(Option.builder().longOpt("svmmodelfile").hasArg()
				.desc("a file to store the SVM model (default=model.mdl)").required(false).build());

	}

	private void runClassification(String args[]) {
		this.configuration = String.join(" ", args);
		DefaultParser parser = new DefaultParser();
		try {
			readOptions(parser.parse(options, args));

		} catch (ParseException e) {
			// TODO Auto-generated catch block

			HelpFormatter formatter = new HelpFormatter();
			System.err.println(e.getMessage());
			formatter.printHelp(
					"java -cp " + (this.getClass().getProtectionDomain().getCodeSource().getLocation().getPath()) + " "
							+ getClass().getCanonicalName().trim() + " [OPTIONS]",
					options);
			return;
		}
	}

	private void readOptions(CommandLine cmd) throws ParseException {

		String inputdirstr = cmd.getOptionValue("inputdirtrain");
		if (inputdirstr != null) {
			inputdir = new File(cmd.getOptionValue("inputdirtrain"));

			if (!inputdir.isDirectory()) {
				throw new ParseException("The mandatory argument 'inputdir' shoud be a directory");
			}
		}

		String inputdirstrtest = cmd.getOptionValue("inputdirtest");
		if (inputdirstrtest != null) {
			inputdirtest = new File(cmd.getOptionValue("inputdirtest"));

			if (!inputdirtest.isDirectory()) {
				throw new ParseException("The mandatory argument 'inputdir' shoud be a directory");
			}
		}

		String ldamodelfilestr = cmd.getOptionValue("svmmodelfile", null);
		if (ldamodelfilestr != null) {
			svmmodelfile = new File(ldamodelfilestr);

			if (svmmodelfile.isDirectory()) {
				throw new ParseException("The optional argument 'ldamodelfile' shoud be a file name");
			}
			if (!svmmodelfile.getAbsoluteFile().getParentFile().exists()) {
				throw new ParseException("Will not be able to write the model, directory does not exist: "
						+ svmmodelfile.getAbsoluteFile().getParentFile());

			}
		} else {
			// tmp.mkdirs();
			// ldamodelfilefile = new File("tmp/model.mdl");
		}

	}

	public static void main(String[] args) throws IOException {

		if (args.length == 1 && args[0].equals("--test")) {
			{
				String argumentline = "--indir testtweets/ --lang de --numthreads 4 --outdir testtweetsclean/ --stem";
				System.out.println("DEMO mode.\n" + argumentline + "\n");
				args = (argumentline).split("\\s+");

			}
		}

		LibSvmClassifier cmodel = new LibSvmClassifier(args);

		// cmodel.runExperiment(args);
		cmodel.runClassificationEX(args);
	}

	public void runClassificationEX(String args[]) throws IOException {
		runClassification(args);
		setParameters();
		// building the train files
		Vector<Double> vy_train = new Vector<Double>();
		Vector<svm_node[]> vx_train = new Vector<svm_node[]>();

		File folders[] = inputdir.listFiles();
		Hashtable<String, Integer> idx = new Hashtable<String, Integer>();

		int i = 1;
		Hashtable<Double, String> labelidx = new Hashtable<>();
		Hashtable<String, Double> invlabelidx = new Hashtable<>();

		for (File f : inputdir.listFiles()) {
			if (!f.isDirectory()) {
				continue;
			}
			double label =  i;
			i++;

			labelidx.put(label, f.getName());
			invlabelidx.put(f.getName(), label);

			for (File inp : f.listFiles()) {

				FileReader fr = new FileReader(inp);
				BufferedReader br = new BufferedReader(fr);
				String line = null;
				while ((line = br.readLine()) != null) {
					svm_node nodeX[] = text2Node(line, idx);
					vx_train.addElement(nodeX);
					vy_train.addElement((double) label);
				}
				fr.close();

			}

		}

		int max_index = idx.size() + 1;

		read_problem(vy_train, vx_train, max_index);
		System.out.println("building the model");
		model = svm.svm_train(prob, param);
		System.out.println("done building the model");
		svm.svm_save_model("textmod.mdl", model);

		Hashtable<Double, Hashtable<Double, Integer>> matrix = new Hashtable<>();

		for (int ii = 1; ii < labelidx.size()+1; ii++) {
			Hashtable<Double, Integer> cn;
			matrix.put((double) ii, cn = new Hashtable<Double, Integer>());

			for (int iy = 1; iy < labelidx.size()+1; iy++) {

				cn.put((double) iy, 0);
			}

		}

		for (File f : inputdirtest.listFiles()) {
			if (!f.isDirectory()) {
				continue;
			}

			for (File inp : f.listFiles()) {
				FileReader fr = new FileReader(inp);
				BufferedReader br = new BufferedReader(fr);
				String line = null;
				while ((line = br.readLine()) != null) {
					svm_node nodeX[] = text2Node(line, idx);

					double predictedValues[] = new double[2];
					double predictedProbabilities[] = new double[2];
					int[] labels = new int[2];

					svm.svm_predict_probability(model, nodeX, predictedProbabilities);
					svm.svm_predict_values(model, nodeX, predictedValues);
					svm.svm_get_labels(model, labels);

					double v = svm.svm_predict(model, nodeX);

					System.out.println(
							"'" + line + "' predicted=" + v + "(" + labelidx.get(v) + ") gt=(" + f.getName() + ")");

					Hashtable<Double, Integer> pred = matrix.get(v);
					Integer curval = pred.get(invlabelidx.get(f.getName()));
					pred.put(invlabelidx.get(f.getName()), curval+1);

				}
				fr.close();

			}
		}
		System.out.print("\t");
		for(double d:matrix.keySet())
		{
			System.out.print(labelidx.get(d)+"\t");
		}
		
		
		for(double d:matrix.keySet())
		{
			Hashtable<Double, Integer> line = matrix.get(d);
			System.out.print("\n"+labelidx.get(d)+"\t");
			for(double d2:line.keySet())
			{
				System.out.print(line.get(d2)+"\t");
			}
		}
		
		System.out.println();

		
		
		
		System.out.println("Preprocessing done");

	}

	public void runExperiment(String args[]) throws IOException {
		runClassification(args);
		// building the train files
		Vector<Double> vy_train = new Vector<Double>();
		Vector<svm_node[]> vx_train = new Vector<svm_node[]>();

		// int count = 0;

		String[] postexts = new String[] { "sonne strand", "sonne strand", "sonne strand", "sonne strand",
				"sonne strand", "sonne strand" };
		String[] negtexts = new String[] { "winter snow", "winter snow", "winter snow", "winter snow", "winter snow",
				"winter snow" };
		String[] nutraltexts = new String[] { "winter strand", "winter snow sonne", "sonne strand strand sonne" };

		Hashtable<String, Integer> idx = new Hashtable<String, Integer>();

		for (String pos : postexts) {
			svm_node nodeX[] = text2Node(pos, idx);
			vx_train.addElement(nodeX);
			vy_train.addElement((double) -1);
		}
		for (String neg : negtexts) {
			svm_node nodeX[] = text2Node(neg, idx);
			vx_train.addElement(nodeX);
			vy_train.addElement((double) 1);
		}

		int max_index = idx.size() + 1;

		// training the model
		setParameters();
		read_problem(vy_train, vx_train, max_index);
		System.out.println("building the model");
		model = svm.svm_train(prob, param);
		System.out.println("done building the model");
		svm.svm_save_model("textmod.mdl", model);

		String[][] texts = new String[][] { postexts, negtexts, nutraltexts };
		int i = 0;
		for (String[] list : texts) {
			System.out.println("texts " + i++);
			for (String neg : list) {
				svm_node nodeX[] = text2Node(neg, idx);

				double predictedValues[] = new double[2];
				double predictedProbabilities[] = new double[2];
				int[] labels = new int[2];

				svm.svm_predict_probability(model, nodeX, predictedProbabilities);
				svm.svm_predict_values(model, nodeX, predictedValues);
				svm.svm_get_labels(model, labels);

				double v = svm.svm_predict(model, nodeX);

				System.out.println("'" + neg + "'=" + v);
			}
		}

	}

	private svm_node[] text2Node(String pos, Hashtable<String, Integer> idx) {
		String[] terms = pos.split("\\s+");

		Hashtable<Integer, svm_node> feature = new Hashtable<Integer, svm_node>();

		for (int i = 0; i < terms.length; i++) {

			Integer dim = null;
			if ((dim = idx.get(terms[i])) == null) {
				idx.put(terms[i], dim = (idx.size() + 1));
			}
			svm_node node = null;
			if ((node = feature.get(dim)) == null) {
				feature.put(dim, node = new svm_node());
				node.index = dim;
				node.value = 0;
			}
			node.value += 1;
		}
		svm_node x[] = new svm_node[feature.size()];
		int cnt = 0;
		for (Integer dim : feature.keySet()) {
			x[cnt++] = feature.get(dim);
		}

		return x;
	}

	private void setParameters() {
		// int i;
		// svm_print_interface print_func = null; // default printing to stdout

		param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.NU_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 3;
		param.gamma = 0; // 1/num_features
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
		// cross_validation = 0;

	}

	public static double atof(String s) {
		double d = Double.valueOf(s).doubleValue();
		if (Double.isNaN(d) || Double.isInfinite(d)) {
			System.err.print("NaN or Infinity in input\n");
			System.exit(1);
		}
		return (d);
	}

	public static int atoi(String s) {
		return Integer.parseInt(s);
	}

	public static void exit_with_help() {
		System.out.print("Usage: svm_train [options] training_set_file [model_file]\n" + "options:\n"
				+ "-s svm_type : set type of SVM (default 0)\n" + "	0 -- C-SVC\n" + "	1 -- nu-SVC\n"
				+ "	2 -- one-class SVM\n" + "	3 -- epsilon-SVR\n" + "	4 -- nu-SVR\n"
				+ "-t kernel_type : set type of kernel function (default 2)\n" + "	0 -- linear: u'*v\n"
				+ "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
				+ "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n" + "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
				+ "	4 -- precomputed kernel (kernel values in training_set_file)\n"
				+ "-d degree : set degree in kernel function (default 3)\n"
				+ "-g gamma : set gamma in kernel function (default 1/num_features)\n"
				+ "-r coef0 : set coef0 in kernel function (default 0)\n"
				+ "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
				+ "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
				+ "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
				+ "-m cachesize : set cache memory size in MB (default 100)\n"
				+ "-e epsilon : set tolerance of termination criterion (default 0.001)\n"
				+ "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
				+ "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
				+ "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
				+ "-v n : n-fold cross validation mode\n" + "-q : quiet mode (no outputs)\n");
		System.exit(1);
	}

	private void read_problem(Vector<Double> vy, Vector<svm_node[]> vx, int max_index) throws IOException {

		prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for (int i = 0; i < prob.l; i++)
			prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for (int i = 0; i < prob.l; i++)
			prob.y[i] = vy.elementAt(i);

		if (param.gamma == 0 && max_index > 0)
			param.gamma = 1.0 / max_index;

		if (param.kernel_type == svm_parameter.PRECOMPUTED)
			for (int i = 0; i < prob.l; i++) {
				if (prob.x[i][0].index != 0) {
					System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
					System.exit(1);
				}
				if ((int) prob.x[i][0].value <= 0 || (int) prob.x[i][0].value > max_index) {
					System.err.print("Wrong input format: sample_serial_number out of range\n");
					System.exit(1);
				}
			}

	}

}
