package com.ds4h.view.reuseGUI;

import com.ds4h.controller.alignmentController.AlignmentControllerInterface;
import com.ds4h.controller.imageController.ImageController;
import com.ds4h.model.alignedImage.AlignedImage;

import javax.swing.*;
import java.awt.*;
import java.util.LinkedList;
import java.util.List;

public class ReuseImagesPanel extends JPanel {

    private final ImageController controller;
    private JPanel currentPanel;
    private final List<PreviewListReuse> listPanels;

    public ReuseImagesPanel(final ImageController controller) {
        this.controller = controller;
        this.listPanels = new LinkedList<>();
        this.setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        this.setVisible(true);
    }

    public void showPreviewImages() {
        this.removeAll();
        this.revalidate();
        for (AlignedImage image : this.controller.getAlignedImages()) {
            final PreviewListReuse panel = new PreviewListReuse(this.controller, image, this);
            panel.setPreferredSize(this.getPreferredSize());
            panel.setAlignmentX(Component.LEFT_ALIGNMENT);
            this.add(panel);
            this.listPanels.add(panel);
        }
        this.revalidate();
    }

    public List<AlignedImage> getImagesToReuse() {
        final List<AlignedImage> images = new LinkedList<>();
        for (final PreviewListReuse innerPanel : this.listPanels) {
            if (innerPanel.toReuse()) {
                images.add(innerPanel.getImage());
            }
        }
        System.out.println(images.size());
        return images;
    }
}
