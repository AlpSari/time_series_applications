# Digital signal processing

Digital Signal Processing (DSP) involves the analysis and manipulation of
**signals**, which are representations of physical quantities. It is used to:

- Filter signals (e.g., remove noise).
- Compress signals (e.g., in audio or video).
- Modify signals (e.g., pitch shifting, equalization).

 A **system** processes input signals to produce output signals. Most DSP applications starts from the digitized signal and processes it by a series of systems.

- Digital signals are obtained by an analog-to-digital converter (ADC). An ADC does the conversion of the analog signal to digital signal by periodically **sampling** the input analog signal.

- Digital systems can either be designed in the digital domain, or be obtained by applying transformations to a system designed in the analog domain.

## Application

Refer to each notebook for different topics and applications on DSP. Recommended reads are:

* [digital_signals](./digital_signals.ipynb): Focuses on the distinction
   between continuous-time (CT) and discrete-time (DT) representations of
   signals.

* [digital_systems](./digital_systems.ipynb): Focuses on the distinction
   between continuous-time (CT) and discrete-time (DT) systems and how to convert a CT LTI system to a DT system.

* [filtering](./filtering.ipynb): Includes applications and properties of DT LTI
  filters. FIR and IIR filters and their comparisons.
