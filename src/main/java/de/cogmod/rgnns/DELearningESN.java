package de.cogmod.rgnns;

import de.cogmod.rgnns.examples.ReservoirToolsExample;
import de.jannlab.optimization.BasicOptimizationListener;
import de.jannlab.optimization.Objective;
import de.jannlab.optimization.optimizer.DifferentialEvolution;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static de.cogmod.rgnns.ReservoirTools.*;
import static de.cogmod.rgnns.ReservoirTools.matrixAsString;

public class DELearningESN {
    public static double sq(final double x) {
        return x * x;
    }

    public static void main(String[] args) {
      
    	//
    	// TODO: Implement DE learning procedure here.
    	//
        //
        // In this example, we use DifferentialEvolution
        // so solve the same least squares problem as in
        // ReservoirToolsExample.
        //
//        final double[][] A = ReservoirToolsExample.A;
//        final double[][] A;
        final double[][] b;// = ReservoirToolsExample.b;
        Random rand = new Random();
        try {
            b = loadSequence("data/sequence.txt");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
//        System.out.println(b.length);
        final double[][] A = new double[b.length][100];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                A[i][j] = rand.nextFloat();
            }
        }

        //
        final int rowsb = rows(b);
        final int colsb = cols(b);
        //
        final int rowsx = cols(A);
        final int colsx = cols(b);
        final double[][] x = new double[rowsx][colsx];
        final int sizex = rowsx * colsx;
        //
        // First, we need an objective (fitness) function that
        // we want optimize (minimize). This can be done by implementing
        // the interface Objective.
        //
        final Objective f = new Objective() {
            //
            @Override
            public int arity() {
                return sizex;
            }
            @Override
            /**
             * This is the callback method that is called from the
             * optimizer to compute the "fitness" of a particular individual.
             */
            public double compute(double[] values, int offset) {
                //
                // the parameters for which the optimizer requests a fitness
                // value or stored in values starting at the given offset
                // with the length that is given via arity(), namely, sizex.
                //
                final double[][] x = new double[rowsx][colsx];
                map(values, offset, x);
                //
                // Compute A * x.
                //
                final double[][] Ax = multiply(A, x);
                //
                // compute square error of x and b.
                //
                double error = 0.0;
                //
                for (int i = 0; i < rowsb; i++) {
                    for (int j = 0; j < colsb; j++) {
                        error += sq(b[i][j] - Ax[i][j]);
                    }
                }
                return error;
            }
        };
        //
        // Now we setup the optimizer.
        //
        final DifferentialEvolution optimizer = new DifferentialEvolution();
        //
        // The same parameters can be used for reservoir optimization.
        //
        optimizer.setF(0.4);
        optimizer.setCR(0.6);
        optimizer.setPopulationSize(5);
        optimizer.setMutation(DifferentialEvolution.Mutation.CURR2RANDBEST_ONE);
        //
        optimizer.setInitLbd(-0.1);
        optimizer.setInitUbd(0.1);
        //
        // Obligatory things...
        //
        optimizer.setRnd(new Random(1234));
        optimizer.setParameters(f.arity());
        optimizer.updateObjective(f);
        //
        // for observing the optimization process.
        //
        optimizer.addListener(new BasicOptimizationListener());
        //
        optimizer.initialize();
        //
        // go!
        //
        optimizer.iterate(1000, 0.0);
        //
        // read the best solution.
        //
        final double[] solution = new double[f.arity()];
        optimizer.readBestSolution(solution, 0);
        map(solution, 0, x);
        //
        // Print out solution. Note that least squares solution is:
        // 0.68
        // -0.07
        // -0.10
        // -0.05
        // 0.69
        //
        System.out.println();
        System.out.println("Evolved solution for Ax = b");
        System.out.println(matrixAsString(x, 2));

    }
    
    
    /**
	 * Helper method for sequence loading from file.
	 */
	public static double[][] loadSequence(final String filename) throws FileNotFoundException, IOException {
        return loadSequence(new FileInputStream(filename));
    }

	/**
	 * Helper method for sequence loading from InputStream.
	 */
    public static double[][] loadSequence(final InputStream inputstream) throws IOException {
        //

        final BufferedReader input = new BufferedReader(
            new InputStreamReader(inputstream));
        //
        final List<String[]> data = new ArrayList<String[]>();
        int maxcols = 0;
        //
        boolean read = true;
        //
        while (read) {
            final String line = input.readLine();
            
            if (line != null) {
                final String[] components = line.trim().split("\\s*(,|\\s)\\s*");
                final int cols = components.length;
                if (cols > maxcols) {
                    maxcols = cols;
                }
                data.add(components);
            } else {
                read = false;
            }
        }
        input.close();
        //
        final int cols = maxcols;
        final int rows = data.size();
        //
        if ((cols == 0) || (rows == 0)) return null;
        //
        final double[][] result = new double[rows][cols];
        //
        for (int r = 0; r < rows; r++) {
            String[] elements = data.get(r);
            for (int c = 0; c < cols; c++) {
                final double value = Double.parseDouble(elements[c]);
                result[r][c] = value;
            }
        }
        //
        return result;
    }
    
    
 
}