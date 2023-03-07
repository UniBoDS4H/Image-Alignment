package com.ds4h.controller.alignmentController.ManualAlignmentController;

import com.ds4h.controller.alignmentController.AlignmentControllerInterface;
import com.ds4h.controller.pointController.PointController;
import com.ds4h.model.alignedImage.AlignedImage;
import com.ds4h.model.alignment.ManualAlgorithm;
import com.ds4h.model.alignment.AlignmentAlgorithmEnum;
import com.ds4h.model.alignment.manual.*;

import java.util.*;

/**
 *
 */
public class ManualAlignmentController implements AlignmentControllerInterface {
    private final ManualAlgorithm perspectiveAlignment;
    private final ManualAlgorithm affineAlignment;
    private final ManualAlgorithm translativeAlignment;
    private final ManualAlgorithm ransacAlignment;
    private final ManualAlgorithm leastMedianAlignment;
    private final List<AlignedImage> images;
    private Optional<AlignmentAlgorithmEnum> lastAlgorithm;
    public ManualAlignmentController(){
        this.images = new LinkedList<>();
        this.lastAlgorithm = Optional.empty();
        this.perspectiveAlignment = new PerspectiveAlignment();
        this.affineAlignment = new AffineAlignment();
        this.translativeAlignment = new TranslationalAlignment();
        this.ransacAlignment = new RansacAlignment();
        this.leastMedianAlignment = null;//new LeastMedianAlignment();
    }

    /**
     *
     * @return
     */
    @Override
    public List<AlignedImage> getAlignedImages(){
        if(this.lastAlgorithm.isPresent()) {

            switch (this.lastAlgorithm.get()){
                case TRANSLATIONAL:
                    return new LinkedList<>(translativeAlignment.alignedImages());
                case AFFINE:
                    return new LinkedList<>(affineAlignment.alignedImages());
                case PROJECTIVE:
                    return new LinkedList<>(perspectiveAlignment.alignedImages());
                case RANSAC:
                    return new LinkedList<>(ransacAlignment.alignedImages());
            }
            this.lastAlgorithm = Optional.empty();
        }
        return Collections.emptyList();
    }

    /**
     *
     * @return
     */
    @Override
    public boolean isAlive() {
        if(this.lastAlgorithm.isPresent()) {
            switch (this.lastAlgorithm.get()) {
                case TRANSLATIONAL:
                    return translativeAlignment.isAlive();
                case AFFINE:
                    return affineAlignment.isAlive();
                case PROJECTIVE:
                    return perspectiveAlignment.isAlive();
                case RANSAC:
                    return ransacAlignment.isAlive();
            }
        }
        return false;
    }

    /**
     *
     * @return
     */
    @Override
    public String name() {
        return this.lastAlgorithm.map(algorithmEnum -> algorithmEnum + " " + "Algorithm").orElse("");
    }

    /**
     * Align manually the images using the Homography alignment.
     * @param cornerManager for each Image we have its own points
     */

    public void alignImages(final AlignmentAlgorithmEnum alignmentAlgorithm, final PointController cornerManager) throws IllegalArgumentException{
        switch (alignmentAlgorithm){
            case TRANSLATIONAL:
                this.align(this.translativeAlignment, cornerManager);
                break;
            case AFFINE:
                this.align(this.affineAlignment, cornerManager);
                break;
            case PROJECTIVE:
                this.align(this.perspectiveAlignment, cornerManager);
                break;
            case RANSAC:
                this.align(this.ransacAlignment, cornerManager);
                break;
        }
        this.lastAlgorithm = Optional.of(alignmentAlgorithm);
        System.out.println(this.lastAlgorithm);
    }

    /**
     *
     * @param algorithm
     * @param cornerManager
     */
    public void align(final ManualAlgorithm algorithm, final PointController cornerManager){
        if(!algorithm.isAlive() && Objects.nonNull(cornerManager) && Objects.nonNull(cornerManager.getCornerManager())) {
            algorithm.alignImages(cornerManager.getCornerManager());
        }
    }
}
