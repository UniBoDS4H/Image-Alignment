/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package com.ds4h.imagej;

import ij.ImagePlus;
import com.ds4h.view.mainGUI.MainMenuGUI;
import ij.plugin.PlugIn;

import java.awt.*;

/**
 * A template for processing each pixel of either
 * GRAY8, GRAY16, GRAY32 or COLOR_RGB images.
 *
 * Authors DS4H Team : Iorio Matteo & Vincenzi Fabio
 */
public class Process_Pixels implements PlugIn {


	public static void main(String[] args) throws Exception {
		new Process_Pixels().run(null);
	}


	@Override
	public void run(String s) {
		EventQueue.invokeLater(MainMenuGUI::new);
	}
}
