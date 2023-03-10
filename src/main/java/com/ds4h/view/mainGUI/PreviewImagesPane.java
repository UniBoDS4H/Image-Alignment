package com.ds4h.view.mainGUI;

import com.ds4h.controller.pointController.PointController;
import com.ds4h.model.imagePoints.ImagePoints;

import javax.swing.*;
import java.awt.*;

public class PreviewImagesPane extends JPanel {
    private final PointController controller;
    private final MainMenuGUI container;
    private final JScrollPane scrollPane;
    JPanel innerPanel;
    PreviewImagesPane(PointController controller, MainMenuGUI container){
        this.container = container;
        this.controller = controller;
        this.scrollPane = new JScrollPane();
        this.innerPanel = new JPanel();
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        this.setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        this.setVisible(true);
    }

    public MainMenuGUI getMainMenu(){
        return this.container;
    }
    public void showPreviewImages(){
        this.removeAll();
        this.revalidate();
        this.innerPanel.removeAll();

        innerPanel.setLayout(new BoxLayout(innerPanel, BoxLayout.Y_AXIS));
        for (final ImagePoints image : this.controller.getCornerImagesImages()) {
            final PreviewListItem panel = new PreviewListItem(controller, image, this, this.controller.getCornerImagesImages().indexOf(image)+1);
            panel.setPreferredSize(this.getPreferredSize());
            panel.setAlignmentX(Component.LEFT_ALIGNMENT);
            panel.setPreferredSize(new Dimension(0,this.getHeight()/6)); // Imposta la dimensione preferita del pannello di anteprima
            innerPanel.add(panel);
        }
        scrollPane.setViewportView(innerPanel);
        this.add(scrollPane);
        this.revalidate();
    }

    public void clearPanels(){
        this.removeAll();
        this.repaint();
    }

    public void updateList(){
        this.showPreviewImages();
        this.repaint();
    }
}
