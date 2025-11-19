

def icontrol_plate_range(id):
    return f"""                </PlateRange>
                <PlateRange id="{id}" range="A1:H12" auto="false">\n""", id+1


def icontrol_header_abs(plate = "GRE96ft",
                        numflashes=3,
                        abs_nm=720,
                        settletime=20,  
                        ):
    icontrol_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<TecanFile xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="tecan.at.schema.documents Main.xsd" fileformat="Tecan.At.Measurement" fileversion="2.0" xmlns="tecan.at.schema.documents">
    <FileInfo type="" instrument="infinite 200Pro" version="" createdFrom="COMPUTER" createdAt="2022-04-26T13:11:50.5003278Z" createdWith="Tecan.At.XFluor.ReaderEditor.XFluorReaderEditor" description="" />
    <TecanMeasurement id="1" class="Measurement">
        <MeasurementManualCycle id="2" number="3" type="Standard">
            <CyclePlate id="3" file="{plate}" plateWithCover="False">
                <PlateRange id="4" range="A1:H12" auto="false">
                    <MeasurementAbsorbance id="5" mode="Normal" type="" name="ABS" longname="" description="">
                        <Well id="6" auto="true">
                            <MeasurementReading id="7" name="" beamDiameter="700" beamGridType="Single" beamGridSize="1" beamEdgeDistance="auto">
                                <ReadingLabel id="8" name="Label1" scanType="ScanFixed" refID="0">
                                    <ReadingSettings number="{numflashes}" rate="25000" />
                                    <ReadingTime integrationTime="0" lagTime="0" readDelay="{settletime}000" flash="0" dark="0" excitationTime="0" />
                                    <ReadingFilter id="0" type="Ex" wavelength="{abs_nm}0" bandwidth="90" attenuation="0" usage="ABS" compatibility="" />
                                </ReadingLabel>
                            </MeasurementReading>
                        </Well>
                        <CustomData id="9" />
                    </MeasurementAbsorbance>\n'''
    return icontrol_header



def icontrol_fluorescent_scan(id,
                              idx,
                              ex,
                              em_range,
                              nm_interval,
                              gain=100,
                              zposition=20000,
                              lagtime=0,
                              integrationtime=20,
                              numflashes=3,
                              settletime=20,
                              ):
    return f"""                    <MeasurementFluoInt readingMode="Bottom" id="{id}" mode="Normal" type="" name="FluoInt" longname="" description="">
                        <Well id="{id+1}" auto="true">
                            <MeasurementReading id="{id+2}" name="" beamDiameter="0" beamGridType="Single" beamGridSize="0" beamEdgeDistance="">
                                <ReadingLabel id="{id+3}" name="Label{idx+2}" scanType="ScanEM" refID="0">
                                    <ReadingSettings number="{numflashes}" rate="25000" />
                                    <ReadingGain type="" gain="{gain}" optimalGainPercentage="0" automaticGain="False" mode="Manual" />
                                    <ReadingTime integrationTime="{integrationtime}" lagTime="{lagtime}" readDelay="{settletime}000" flash="0" dark="0" excitationTime="0" />
                                    <ReadingFilter id="{id+4}" type="Ex" wavelength="{ex}0" bandwidth="50" attenuation="0" usage="FI" compatibility="" />
                                    <ReadingFilter id="{id+5}" type="Em" wavelength="{em_range.min()}0~{em_range.max()}0:{nm_interval}0" bandwidth="200" attenuation="0" usage="FI" compatibility="" />
                                </ReadingLabel>
                            </MeasurementReading>
                        </Well>
                    </MeasurementFluoInt>\n""", id+5

def icontrol_end(id, 
                 idx,
                 numflashes=3,
                 abs_nm=720,
                 settletime=20):
    return f"""                </PlateRange>
                <PlateRange id="{id}" range="A1:H12" auto="false">
                    <MeasurementAbsorbance id="{id+1}" mode="Normal" type="" name="ABS" longname="" description="">
                        <Well id="{id+2}" auto="true">
                            <MeasurementReading id="{id+3}" name="" beamDiameter="700" beamGridType="Single" beamGridSize="1" beamEdgeDistance="auto">
                                <ReadingLabel id="{id+4}" name="Label{idx+3}" scanType="ScanFixed" refID="0">
                                    <ReadingSettings number="{numflashes}" rate="25000" />
                                    <ReadingTime integrationTime="0" lagTime="0" readDelay="{settletime}000" flash="0" dark="0" excitationTime="0" />
                                    <ReadingFilter id="0" type="Ex" wavelength="{abs_nm}0" bandwidth="90" attenuation="0" usage="ABS" compatibility="" />
                                </ReadingLabel>
                            </MeasurementReading>
                        </Well>
                        <CustomData id="{id+5}" />
                    </MeasurementAbsorbance>
                </PlateRange>
            </CyclePlate>
        </MeasurementManualCycle>
        <MeasurementInfo id="0" description="">
            <ScriptTemplateSettings id="0">
                <ScriptTemplateGeneralSettings id="0" Title="" Group="" Info="" Image="" />
                <ScriptTemplateDescriptionSettings id="0" Internal="" External="" IsExternal="False" />
            </ScriptTemplateSettings>
        </MeasurementInfo>
    </TecanMeasurement>
</TecanFile>"""

