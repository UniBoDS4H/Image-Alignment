package com.ds4h.view.mainGUI;
import com.ds4h.controller.alignmentController.AlignmentControllerInterface;
import com.ds4h.controller.alignmentController.AutomaticAlignmentController.AutomaticAlignmentController;
import com.ds4h.controller.alignmentController.ManualAlignmentController.ManualAlignmentController;
import com.ds4h.controller.alignmentController.semiAutomaticController.SemiAutomaticController;
import com.ds4h.controller.bunwarpJController.BunwarpJController;
import com.ds4h.controller.changeColorController.ChangeColorController;
import com.ds4h.controller.pointController.PointController;
import com.ds4h.controller.directoryManager.DirectoryManager;
import com.ds4h.controller.exportController.ExportController;
import com.ds4h.controller.imageController.ImageController;
import com.ds4h.controller.importController.ImportController;
import com.ds4h.controller.opencvController.OpencvController;
import com.ds4h.model.alignment.alignmentAlgorithm.*;
import com.ds4h.model.imagePoints.ImagePoints;
import com.ds4h.view.aboutGUI.AboutGUI;
import com.ds4h.view.alignmentConfigGUI.AlignmentConfigGUI;
import com.ds4h.view.automaticSettingsGUI.AutomaticSettingsGUI;
import com.ds4h.view.bunwarpjGUI.BunwarpjGUI;
import com.ds4h.view.carouselGUI.CarouselGUI;
import com.ds4h.view.displayInfo.DisplayInfo;
import com.ds4h.view.loadingGUI.LoadingGUI;
import com.ds4h.view.overlapImages.OverlapImagesGUI;
import com.ds4h.view.standardGUI.StandardGUI;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.process.ImageProcessor;

import javax.swing.*;
import javax.swing.event.MouseInputListener;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Collectors;

import static com.ds4h.model.util.AlignmentUtil.getAlgorithmFromEnum;


public class MainMenuGUI extends JFrame implements StandardGUI {
    private final JButton manualAlignment, automaticAlignment;
    private final JMenuBar menuBar;
    private final JMenu menu, project, settings, about;
    private final JMenuItem settingsItem, loadImages, exportItem, importItem, clearItem, alignmentItem, automaticItem;
    private final JPanel panel;
    private final AboutGUI aboutGUI;
    private final JFileChooser fileChooser;
    private final BunwarpjGUI settingsBunwarpj;
    private final PointController cornerControler;
    private final PreviewImagesPane imagesPreview;
    private final AlignmentConfigGUI alignmentConfigGUI;
    private final BunwarpJController bunwarpJController;
    private final AutomaticAlignmentController automaticAlignmentController = new AutomaticAlignmentController();
    private final ManualAlignmentController manualAlignmentController = new ManualAlignmentController();
    private final SemiAutomaticController semiAutomaticController = new SemiAutomaticController();
    private final AutomaticSettingsGUI automaticSettingsGUI;

    private static final int MIN_IMAGES = 0, MAX_IMAGES = 3;

    /**
     * Constructor of the MainMenu GUI
     */
    public MainMenuGUI() {
        setTitle("DS4H Image Alignment");
        this.setFrameSize();
        this.bunwarpJController = new BunwarpJController();
        this.fileChooser = new JFileChooser();
        this.cornerControler = new PointController();
        //Init of the two buttons
        this.manualAlignment = new JButton("Manual Alignment");
        this.automaticAlignment = new JButton("Automatic Alignment");

        //Adding the Left Panel, where are stored the buttons for the transformations
        this.panel = new JPanel();
        this.panel.setLayout(new GridBagLayout());

        //Init of the previewList
        this.imagesPreview = new PreviewImagesPane(this.cornerControler, this);

        GridBagConstraints gbcPanel = new GridBagConstraints();
        gbcPanel.gridx = 0;
        gbcPanel.gridy = 0;
        gbcPanel.gridwidth = GridBagConstraints.REMAINDER;
        gbcPanel.gridheight = 10;
        gbcPanel.fill = GridBagConstraints.BOTH;
        gbcPanel.weightx = 1;
        gbcPanel.weighty = 1;
        this.panel.add(new JScrollPane(this.imagesPreview), gbcPanel);

        GridBagConstraints gbcAuto = new GridBagConstraints();
        gbcAuto.gridx = 0;
        gbcAuto.gridy = 10;
        gbcAuto.gridwidth = 1;
        gbcAuto.gridheight = 1;
        gbcAuto.fill = GridBagConstraints.BOTH;
        gbcAuto.weightx = 1;
        gbcAuto.weighty = 0;
        this.panel.add(this.automaticAlignment, gbcAuto);
        
        GridBagConstraints gbcManual = new GridBagConstraints();
        gbcManual.gridx = 0;
        gbcManual.gridy = 11;
        gbcManual.gridwidth = 1;
        gbcManual.gridheight = 1;
        gbcManual.fill = GridBagConstraints.BOTH;
        gbcManual.weightx = 1;
        gbcManual.weighty = 0;
        this.panel.add(this.manualAlignment, gbcManual); // aggiungo il secondo bottone al JFrame con il GridBagLayout

        add(this.panel);
        //this.canvas = new ImageCanvas(new ImagePlus("my stack", this.stack));

        this.aboutGUI = new AboutGUI();
        this.settingsBunwarpj = new BunwarpjGUI(this.bunwarpJController);
        this.alignmentConfigGUI = new AlignmentConfigGUI(this);
        this.automaticSettingsGUI = new AutomaticSettingsGUI(this);
        //Init of the Menu Bar and all the Menu Items
        this.menuBar = new JMenuBar();
        this.menu = new JMenu("File");

        this.project = new JMenu("Project");
        this.about = new JMenu("About");
        this.settings = new JMenu("Settings");
        this.settingsItem = new JMenuItem("BunwarpJ");
        this.loadImages = new JMenuItem("Load Images");
        this.exportItem = new JMenuItem("Export");
        this.importItem = new JMenuItem("Import");
        this.clearItem = new JMenuItem("Clear");
        this.alignmentItem = new JMenuItem("Manual");
        this.automaticItem = new JMenuItem("Automatic");
        this.addComponents();
        this.addListeners();
        this.checkPointsForAlignment();
        this.showDialog();
    }

    /**
     * Add all the components of the MainMenu
     */
    @Override
    public void addComponents(){
        // Create menu bar and add it to the frame
        this.setJMenuBar(this.menuBar);
        // Create menu and add it to the menu bar
        this.menuBar.add(this.menu);
        this.menuBar.add(this.project);
        this.menuBar.add(this.settings);
        this.menuBar.add(this.about);
        // Create menu items and add them to the menu
        this.menu.add(this.loadImages);
        this.settings.add(this.alignmentItem);
        this.settings.add(this.automaticItem);
        this.settings.add(this.settingsItem);
        this.project.add(this.exportItem);
        this.project.add(this.importItem);
        this.project.add(this.clearItem);
    }
    public void checkPointsForAlignment() {
        ToolTipManager.sharedInstance().setDismissDelay(Integer.MAX_VALUE);
        int nPoints;
        if (!this.cornerControler.getCornerImagesImages().isEmpty()) {
            nPoints = this.cornerControler.getCornerImagesImages().get(0).getPoints().length;
            switch (this.alignmentConfigGUI.getSelectedValue()) {
                case TRANSLATIONAL:
                    boolean ok = true;
                    for (ImagePoints i : this.cornerControler.getCornerImagesImages()) {
                        System.out.println(i.getPoints().length);
                        if (i.getPoints().length < TranslationalAlignment.LOWER_BOUND) {
                            ok = false;
                            break;
                        }
                    }
                    if(ok){
                        for (ImagePoints i : this.cornerControler.getCornerImagesImages()) {
                            if (i.getPoints().length != nPoints) {
                                ok = false;
                                break;
                            }
                        }
                        this.manualAlignment.setEnabled(ok);
                        this.manualAlignment.setToolTipText(ok?"":"The number of points inside the images is not the same in all of them.");
                    }else{
                        this.manualAlignment.setEnabled(false);
                        this.manualAlignment.setToolTipText("<html>"
                                + "The number of points inside the images is not correct."
                                + "<br>"
                                + "In order to use the Translation alignment you must use at least " + TranslationalAlignment.LOWER_BOUND + " points in each image."
                                + "</html>");
                    }
                    break;
            }
        }
    }

    @Override
    public void showDialog() {
        setVisible(true);
    }

    /**
     * Add all the listeners to the components of the MainMenu
     */
    @Override
    public void addListeners() {
        // Add event listener to the menu items
        this.about.addMouseListener(new MouseInputListener() {
            @Override
            public void mouseClicked(MouseEvent e) {

            }

            @Override
            public void mousePressed(MouseEvent e) {
                System.out.println("ciao");
                aboutGUI.showDialog();
            }

            @Override
            public void mouseReleased(MouseEvent e) {

            }

            @Override
            public void mouseEntered(MouseEvent e) {

            }

            @Override
            public void mouseExited(MouseEvent e) {

            }

            @Override
            public void mouseDragged(MouseEvent e) {

            }

            @Override
            public void mouseMoved(MouseEvent e) {

            }
        });

        this.clearItem.addActionListener(event -> {
            final int result = JOptionPane.showConfirmDialog(this,
                    "Are you sure to clear the entire project ?",
                    "Confirm operation",
                    JOptionPane.YES_NO_OPTION);
            if(result == JOptionPane.YES_OPTION) {
                this.cornerControler.clearProject();
                this.imagesPreview.clearPanels();
                this.imagesPreview.showPreviewImages();
            }
        });

        this.loadImages.addActionListener(event ->{
            this.pickImages();
        });

        this.settingsItem.addActionListener(event ->{
            this.settingsBunwarpj.showDialog();
        });

        this.manualAlignment.addActionListener(event -> {
            this.pollingManualAlignment();
        });

        this.alignmentItem.addActionListener(event -> {
            this.alignmentConfigGUI.showDialog();
        });

        this.automaticItem.addActionListener(event -> {
            this.automaticSettingsGUI.showDialog();
        });

        this.exportItem.addActionListener(event -> {
            this.fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            final int result = this.fileChooser.showOpenDialog(this);
            if(result == JFileChooser.APPROVE_OPTION){
                final File file = this.fileChooser.getSelectedFile();
                try {
                    ExportController.exportProject(this.cornerControler.getCornerManager(), file.getPath());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });

        this.importItem.addActionListener(event -> {
            this.fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            final int result = this.fileChooser.showOpenDialog(this);
            if(result == JFileChooser.APPROVE_OPTION){
                final File file = this.fileChooser.getSelectedFile();
                try {
                    ImportController.importProject(file, this.cornerControler.getCornerManager());
                    this.imagesPreview.showPreviewImages();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        });

        this.automaticAlignment.addActionListener(event -> {
            this.pollingAutomaticAlignment();
        });

        this.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                DirectoryManager.deleteTMPDirectories();
                OpencvController.deleteLibrary();
                dispose();
            }
        });
    }

    private void pollingManualAlignment(){
        if(!this.manualAlignmentController.isAlive()) {
            try {
                AlignmentAlgorithm alg = getAlgorithmFromEnum(this.alignmentConfigGUI.getSelectedValue());
                if(alg instanceof  TranslationalAlignment){
                    ((TranslationalAlignment) alg).setTransformation(this.alignmentConfigGUI.getTranslation(),this.alignmentConfigGUI.getRotation(), this.alignmentConfigGUI.getScaling());
                }
                this.manualAlignmentController.align(alg, this.cornerControler);
                this.startPollingThread(this.manualAlignmentController);
            }catch(final Exception e){
                JOptionPane.showMessageDialog(this,
                        e.getMessage(),
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void startPollingThread(final AlignmentControllerInterface alignmentControllerInterface){
        final Thread pollingSemiautomaticAlignment = new Thread(() -> {
            final LoadingGUI loadingGUI = new LoadingGUI();
            while (alignmentControllerInterface.isAlive()) {
                try {
                    Thread.sleep(2000);
                } catch (final InterruptedException e) {
                    JOptionPane.showMessageDialog(this,
                            e.getMessage(),
                            "Error",
                            JOptionPane.ERROR_MESSAGE);
                    loadingGUI.close();
                    return;
                }
            }
            if (alignmentControllerInterface.getAlignedImages().size() > 0) {
                if(alignmentControllerInterface instanceof ManualAlignmentController) {
                    new CarouselGUI(alignmentControllerInterface.name(), this.settingsBunwarpj, new ImageController(alignmentControllerInterface, bunwarpJController), this.cornerControler, this.imagesPreview);

                }else{
                    final OverlapImagesGUI overlapImagesGUI = new OverlapImagesGUI(alignmentControllerInterface.name(),this.settingsBunwarpj, new ImageController(alignmentControllerInterface, bunwarpJController), this.cornerControler, this.imagesPreview);
                    overlapImagesGUI.showDialog();
                }
                loadingGUI.close();
            }
        });
        pollingSemiautomaticAlignment.start();
    }

    private void pollingAutomaticAlignment(){
        if(!this.automaticAlignmentController.isAlive()) {
            try {
                this.automaticAlignmentController.align(this.automaticSettingsGUI.getSelectedDetector(), this.cornerControler);
                this.startPollingThread(this.automaticAlignmentController);
            }catch (final Exception e){
                JOptionPane.showMessageDialog(this,
                        e.getMessage(),
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    /**
     * Open a File dialog in order to choose all the images for our tool
     */
    private void pickImages(){
        final FileDialog fd = new FileDialog(new Frame(), "Choose files", FileDialog.LOAD);
        fd.setMultipleMode(true);
        fd.setVisible(true);
        final File[] files = fd.getFiles();//Get all the files
        try {
            this.cornerControler.loadImages(Arrays.stream(files).map(File::getPath).collect(Collectors.toList()));
            this.imagesPreview.showPreviewImages();
        }catch (final Exception exception){
            JOptionPane.showMessageDialog(this,
                    exception.getMessage(),
                    "Error",
                    JOptionPane.ERROR_MESSAGE);
        }
    }

    /**
     * Method used to set the Min dimension of the Frame, based on the Users monitor dimension
     */
    private void setFrameSize(){
        // Get the screen size
        final Dimension screenSize = DisplayInfo.getDisplaySize(80);
        final int min_width = (int) (screenSize.width/5);
        final int min_height =(int) (screenSize.height);
        // Set the size of the frame to be half of the screen width and height
        // Set the size of the frame to be half of the screen width and height
        setSize(min_width, min_height);
        setMinimumSize(new Dimension(min_width,min_height));
    }

    public void reloadImages(){
        this.imagesPreview.showPreviewImages();
    }

}