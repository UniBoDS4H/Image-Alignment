package com.ds4h.view.exportGUI;

import com.ds4h.controller.alignmentController.AlignmentControllerInterface;
import com.ds4h.controller.exportController.ExportController;
import com.ds4h.model.alignedImage.AlignedImage;
import com.ds4h.model.util.Pair;
import com.ds4h.view.bunwarpjGUI.BunwarpjGUI;
import com.ds4h.view.displayInfo.DisplayInfo;
import com.ds4h.view.standardGUI.StandardGUI;
import ij.IJ;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class ExportZipGUI extends JFrame {
    /*
    private final JButton button;
    private final AlignmentControllerInterface controllerInterface;
    private final JList<Panel> images;
    private final JTextField textField;
    private final JCheckBox includeImage;
    private final JFileChooser fileChooser;
    private final GridBagConstraints constraints;



    public ExportZipGUI(final AlignmentControllerInterface controller){

        this.setLayout(new GridBagLayout());
        // Set the Frame size
        this.setSize();
        this.button = new JButton("Save");
        this.controllerInterface = controller;
        this.constraints = new GridBagConstraints();
        this.constraints.insets = new Insets(0, 0, 5, 5);
        this.constraints.anchor = GridBagConstraints.WEST;
        this.textField = new JTextField();

        this.includeImage = new JCheckBox();
        final List<JPanel> ps = new ArrayList<>();
        this.controllerInterface.getAlignedImages().forEach(image -> {
            final JPanel p = new JPanel();
            p.add(new JTextField("AOO"));
            p.add(new JCheckBox("DIO CANE"));
            ps.add(p);
        });
        this.images = new JList<>();
        this.fileChooser = new JFileChooser();
        this.addComponents();
        this.addListeners();

    }

    private void setSize(){
        Dimension screenSize = DisplayInfo.getDisplaySize(50);
        int min_width = (int) (screenSize.width);
        int min_height =(int) (screenSize.height);
        // Set the size of the frame to be half of the screen width and height
        // Set the size of the frame to be half of the screen width and height
        setSize(min_width, min_height);
        setMinimumSize(new Dimension(min_width,min_height));
    }

    private void setFrameSize(){
        final Dimension newDimension = DisplayInfo.getDisplaySize(50);
        this.setSize((int)newDimension.getWidth(), (int)newDimension.getHeight());
        this.setMinimumSize(newDimension);
    }

    @Override
    public void showDialog() {
        this.setVisible(true);
    }

    @Override
    public void addListeners() {
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        this.button.addActionListener(event -> {
            final int result = fileChooser.showDialog(this, "Select a Directory");
            if(result == JFileChooser.APPROVE_OPTION){
                File selectedFile = fileChooser.getSelectedFile();
                String path = selectedFile.getAbsolutePath();
                try {
                    ExportController.exportAsZip(this.controllerInterface.getAlignedImages(), path);
                } catch (IOException e) {
                    IJ.showMessage(e.getMessage());
                }
            }
        });
    }

    @Override
    public void addComponents() {
        this.fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        this.addElement(new JLabel("Select the image"), new JPanel(), this.images);
        this.addElement(new JLabel("Change the name"), new JPanel(), this.textField);
        this.addElement(new JLabel("Include this file ?"), new JPanel(), this.includeImage);
        this.constraints.gridx = 0;
        this.constraints.gridy++;
        add(this.button, this.constraints);
    }

    private void addElement(final JLabel label, final JPanel panel, final JComponent component){
        panel.add(label);
        panel.add(component);
        this.constraints.gridx = 0;
        this.constraints.gridy++;
        add(panel, this.constraints);
    }
    */
}
