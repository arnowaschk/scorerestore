# Architecture

ScoreRestore V1 is a grayscale, tiled restoration and semantic-analysis proof of concept. It is
not an optical music-recognition system: it does not infer pitches, durations, MusicXML, or MIDI.

```text
input PDF / raster → grayscale page → overlap-blended tiles → shared encoder/decoder
                                                        ├─ ink-coverage cleaning head
                                                        └─ four independent semantic heads
```

The readable in-repository U-Net is the primary model: four encoder levels, a bottleneck, bilinear
upsampling, skip connections, GroupNorm, and separate 1-channel cleaning and 4-channel segmentation
heads. The transfer comparison replaces the encoder with ImageNet-pretrained ResNet-18 after
averaging its first RGB kernels to one grayscale input channel. Small tile batches retain trainable
BatchNorm affine parameters while freezing encoder running statistics.

Semantic channels are independent sigmoid probabilities in fixed order:
`background`, `staff`, `notation`, `text`. Foreground channels may overlap—e.g. a notehead can cross
a staff line—so a softmax would discard valid geometry. Background is trained independently and the
evaluation report diagnoses background/foreground overlap and all-false pixels.

At inference, full pages are never resized for the model. Reflection-padded tiles are processed at
bounded memory and their logits are blended with a raised-cosine window before thresholding. The
result preserves original raster dimensions exactly.
