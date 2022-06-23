package de.cogmod.rgnns;

import java.util.Arrays;

/**
 * @author Sebastian Otte
 */
public class EchoStateNetwork extends RecurrentNeuralNetwork {

    private double[][] inputweights;
    private double[][] reservoirweights;
    private double[][] outputweights;
    
    public double[][] getInputWeights() {
        return this.inputweights;
    }
    
    public double[][] getReservoirWeights() {
        return this.reservoirweights;
    }
    
    public double[][] getOutputWeights() {
        return this.outputweights;
    }
    
    public EchoStateNetwork(
        final int input,
        final int reservoirsize,
        final int output
    ) {
        super(input, reservoirsize, output);
        //
        this.inputweights     = this.getWeights()[0][1];
        this.reservoirweights = this.getWeights()[1][1];
        this.outputweights    = this.getWeights()[1][2];
        //
    }
    
    @Override
    public void rebufferOnDemand(int sequencelength) {
        super.rebufferOnDemand(1);
    }
    

    public double[] output() {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer(); 
        //
        final int n = act[outputlayer].length;
        //
        final double[] result = new double[n];
        final int t = Math.max(0, this.getLastInputLength() - 1);
        //
        for (int i = 0; i < n; i++) {
            result[i] = act[outputlayer][i][t];
        }
        //
        return result;
    }
    
    /**
     * This is an ESN specific forward pass realizing 
     * an oscillator by means of an output feedback via
     * the input layer. This method requires that the input
     * layer size matches the output layer size. 
     */
    public double[] forwardPassOscillator() {
        //
        // this method causes an additional copy operation
        // but it is more readable from outside.
        //
        final double[] output = this.output();
        return this.forwardPass(output);
    }
    
    /**
     * Overwrites the current output with the given target.
     */
    public void teacherForcing(final double[] target) {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer();
        //
        final int n = act[outputlayer].length;
        //
//        final int t = this.getLastInputLength()-1;
        final int t = this.getLastInputLength();
        //
        for (int i = 0; i < n; i++) {
            act[outputlayer][i][t] = target[i];
        }
    }
    
    /**
     * ESN training algorithm. 
     */
    public double trainESN(
        final double[][] sequence,
        final int washout,
        final int training,
        final int test
    ) {
        //
        // TODO: implement ESN training algorithm here. 
        //

        // load training data
        de.cogmod.spacecombat.Serializer serializer = new de.cogmod.spacecombat.Serializer();

        // Washout phase
        final EchoStateNetwork esn = new EchoStateNetwork(1, 5, 1);
        double[][] washoutTrajectory;
        washoutTrajectory = Arrays.copyOfRange(sequence, 0, washout);
        for (int i = 0; i < washout; i++) {
            esn.teacherForcing(sequence[0]);
        }
        // Training phase
        double[][] targetSequence;
        double[][] reservoirStates;
        // load collected data
        targetSequence = Arrays.copyOfRange(sequence, washout, washout+training);
        reservoirStates = Arrays.copyOfRange(
                serializer.loadFile("data/reservoirStatesTrain.txt"),
                washout,
                washout+training);
        // convert target sequence and reservoir states to matrices
        org.ejml.simple.SimpleMatrix T = new org.ejml.simple.SimpleMatrix(targetSequence);
        org.ejml.simple.SimpleMatrix M = new org.ejml.simple.SimpleMatrix(reservoirStates);
        // invert M
        org.ejml.simple.SimpleMatrix invM = new org.ejml.simple.SimpleMatrix(M.pseudoInverse());
        // multiply inverted M with T to create W_out
        org.ejml.simple.SimpleMatrix outW = invM.mult(T);

        // parse weights into array and save them
        double[][] weightsTrained = new double[outW.numCols()][outW.numRows()];
        for (int column = 0; column < outW.numCols(); column++) {
            for (int row = 0; row < outW.numRows(); row++) {
                weightsTrained[column][row] = outW.get(row, column);
            }
        }
        serializer.saveFile(weightsTrained, "esn.weights");

        return 0.0; // error.
    }
    
}