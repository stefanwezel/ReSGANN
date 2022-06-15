package de.cogmod.spacecombat;

import java.io.IOException;
import java.util.Random;

/**
 * @author Sebastian Otte
 */
public class Serializer {

	public static void main(String[] args) {
//		TODO add method 'save'

    	final double[] values = new double[10];
    	final Random rnd = new Random(1234);
    	
    	for (int i = 0; i < values.length; i++) {
    		values[i] = rnd.nextDouble();
    	}
    	
    	//
    	// Write double array into file.
    	//
		//		TODO add method 'load'

		try {
			de.jannlab.io.Serializer.write(values, "data/test.dat");
		} catch (IOException e) {
			e.printStackTrace();
		}
		//
		// Read double array from file.
		//
		try {
			final double[] values2 = de.jannlab.io.Serializer.read("data/test.dat");
			
			for (int i = 0; i < values2.length; i++) {
				System.out.println(values2[i]);
			}
			
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
    	System.out.println();

    }  
}