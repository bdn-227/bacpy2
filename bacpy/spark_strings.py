
import numpy as np

def spark_header_abs(wells="all",
                     numflashes=3,
                     abs_nm=720,
                     settletime=20,
                     num_measurements=1):

    string ='<MethodStrip IsApp="False" Version="2" xmlns="http://schemas.tecan.com/at/dragonfly/operations/xaml" '
    string+='xmlns:sco="clr-namespace:System.Collections.ObjectModel;assembly=System" '
    string+='xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"><MethodStrip.DataLabels>'
    
    for e in range(num_measurements+2):
        string+=f'<x:Reference>__ReferenceID{e+2}</x:Reference>'

    string+='</MethodStrip.DataLabels><MethodStrip.Plates><x:Reference>__ReferenceID1</x:Reference>'
    string+='</MethodStrip.Plates><InstrumentStrip><PlateStrip SpillingUserValue="{x:Null}" '
    string+='DefaultMovementSpeed="NORMAL" HumidityCassetteType="None" IsReadBarcode="False" '
    string+='LidType="None" Spilling="False" SpillingSwSource="False" UseBarcode="False">'
    string+='<PlateStrip.DataLabels><sco:ObservableCollection x:TypeArguments="IDataLabel" />'
    string+='</PlateStrip.DataLabels><PlateStrip.Plates><Plate x:Name="__ReferenceID1" Columns="12" '
    string+='PlateNumber="1" Rows="8"><Plate.MicroplateDefinition>'
    string+='<MicroplateDefinition DrawingNumber="{x:Null}" ManufacturerRevisionOfTechnicalDrawing="{x:Null}" TecanCatalogNumber="{x:Null}" Version="{x:Null}" CellImaging="False" Comment="Cat. No.: 655101/655161/655192" CreationDate="2007-09-17T09:45:27.744789Z" DisplayName="[GRE96ft] - Greiner 96 Flat Transparent" IsReadOnlyTemplate="True" Manufacturer="Greiner" Material="Polystyrene" Name="GRE96ft" PlateColor="Transparent" PlateType="StandardMTP" RefractiveIndex="1.5">'
    string+='<MicroplateDefinition.MeasurementAreaDefinitions>'
    string+='<GridMeasurementPosition CoordinateX="14380" CoordinateY="11240" NumberOfColumns="12" '
    string+='NumberOfRows="8" Well2WellCenterSpacingXAxis="9000" '
    string+='Well2WellCenterSpacingYAxis="9000" />'
    string+='</MicroplateDefinition.MeasurementAreaDefinitions>'
    string+='<MicroplateDefinition.PlateFootprintDimension>'
    string+='<PlateFootprintDimension OutsideXTolerance="{x:Null}" OutsideYTolerance="{x:Null}" FlangeHeight="2500" Height="14600" HeightTolerance="200" HeightWithCover="17606" OutsideX="127760" OutsideY="85480"/>'
    string+='</MicroplateDefinition.PlateFootprintDimension>'
    string+='<MicroplateDefinition.WellFootprintDimension>'
    string+='<WellFootprintDimension GrowthArea="{x:Null}" BottomColor="None" BottomDimensionX="6390" BottomDimensionY="6390" BottomShape="Flat" BottomThickness="0" Depth="10900" MaximumCapacity="382" TopDimensionX="6960" TopDimensionY="6960" TopShape="Round" WorkingCapacity="200"/>'
    string+='</MicroplateDefinition.WellFootprintDimension></MicroplateDefinition>'
    string+='</Plate.MicroplateDefinition><Plate.PlateLayout><x:Reference>__ReferenceID0'
    string+='</x:Reference>'

    

    if wells=="all":
        # define all wells
        dfId = [let+str(num) for let in 'ABCDEFGH' for num in range(1,13)][1:]
        indices = list(range(1, len(dfId)+1))
        rows = [np.where(well[0] == np.array(list('ABCDEFGH')))[0][0] for well in dfId]
        cols = [well[1:] for well in dfId]
        for well, row, col, idx in zip(dfId, rows, cols, indices):
            string+=f'<Well CartesianCoordinate="{{x:Null}}" Color="{{x:Null}}" ExperimentalGroup="{{x:Null}}" IdentifierGroupMember="{{x:Null}}" IdentifierReplicate="{{x:Null}}" IdentifierReplicates="{{x:Null}}" AlphanumericCoordinate="{well}" Column="{int(col)-1}" Grid="0" IdentifierGroup="None" IsFlagged="False" IsOut="False" IsSelected="True" Row="{row}" WellIndex="{idx}" />'
        
    string+='</Plate.PlateLayout></Plate></PlateStrip.Plates>'

    # now add absorbance
    string+='<AbsorbanceStrip SelectedInputData="{x:Null}" MeasurementsCount="1" '
    string+=f'MultipleReadsPerWell="False" NumberFlashes="{numflashes}" NumberOfMRWPoints="0" '
    string+='PathlengthCorrectionFactor="0.186" Reference="False" SelectedBorder="500" '
    string+='SelectedMultipleReadsPerWell="NotDefined" '
    string+='SelectedPathLengthCorrectionFactorType="Manual" '
    string+='SelectedPathLengthStatus="NotDefined" SelectedPattern="Square" '
    string+=f'SelectedSize="2" SettleTime="{settletime}" TestWavelength="9770" '
    string+=f'WavelengthMeasurement="{abs_nm}0" WavelengthReference="6200">'
    string+='<AbsorbanceStrip.DataLabels><DataLabel InternalSuffix="{x:Null}" '
    string+='x:Name="__ReferenceID2" Index="0" MeasureMode="SinglePoint" '
    string+='OutputName="Label 1" Type="Measurement" '
    string+='Unit="OpticalDensity" /></AbsorbanceStrip.DataLabels></AbsorbanceStrip>'

    return string


def spark_fluorescence_scan(id,
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
    
    if idx == 0:
        string =f'<FluorescenceIntensityScanStrip CalculatedGainPercentValue="100" Flashes="{numflashes}" '
    else:
        string =f'<FluorescenceIntensityScanStrip GainWell="{{x:Reference __ReferenceID0}}" CalculatedGainPercentValue="100" Flashes="{numflashes}" '
    string+=f'GainMode="Manual" IntegrationTime="{integrationtime}" LagTime="{lagtime}" MeasurementDirection="Bottom" '
    string+='MrwOptimalFlashes="0" MultipleReadsPerWellBorderSize="500" '
    string+='MultipleReadsPerWellMode="NotDefinied" MultipleReadsPerWellPattern="Single" '
    string+='MultipleReadsPerWellPatternSize="0" NumberOfMRWPoints="0" ScanMode="EmissionScan" '
    string+=f'SelectedFluorophoreName="Other" SettleTime="{settletime}000" SignalGain="{gain}" '
    string+='ThreeDimScanGainEmWavelength="5350" ThreeDimScanGainExWavelength="4850" '
    string+='UseGainRegulation="False"><FluorescenceIntensityScanStrip.BottomMirror>'
    string+='<FiMirrorConfig EmissionEndWavelength="9000" EmissionStartWavelength="2300" '
    string+='ExcitationEndWavelength="9000" ExcitationStartWavelength="2300" IsAutomatic="False" '
    string+='IsUserDefinable="False" MirrorType="Bottom" Name="Bottom" />'
    string+='</FluorescenceIntensityScanStrip.BottomMirror>'
    string+='<FluorescenceIntensityScanStrip.DataLabels><DataLabel InternalSuffix="{x:Null}" '
    string+=f'x:Name="__ReferenceID{id+1}" Index="{id-1}" MeasureMode="Scan" OutputName="Label {id}" '
    string+='Type="Measurement" Unit="RelativeFluorescenceUnit" />'
    string+='</FluorescenceIntensityScanStrip.DataLabels>'
    string+='<FluorescenceIntensityScanStrip.EmissionMode><FiWavelengthConfig '
    string+='SelectedFilter="{x:Null}" SelectedFiMode="Monochromator">'
    string+='<FiWavelengthConfig.MonochromatorConfig><FiMonochromatorConfig '
    string+='Bandwidth="200" From="2800" Step="10" To="9000" Wavelength="5450" />'
    string+='</FiWavelengthConfig.MonochromatorConfig></FiWavelengthConfig>'
    string+='</FluorescenceIntensityScanStrip.EmissionMode>'
    string+='<FluorescenceIntensityScanStrip.EmissionScanConfig><FiScanSettings '
    string+=f'Bandwidth="200" From="{em_range.min()}0" Step="{nm_interval}0" To="{em_range.max()}0" />'
    string+='</FluorescenceIntensityScanStrip.EmissionScanConfig>'
    string+='<FluorescenceIntensityScanStrip.ExcitationMode><FiWavelengthConfig '
    string+='SelectedFilter="{x:Null}" SelectedFiMode="Monochromator">'
    string+='<FiWavelengthConfig.MonochromatorConfig><FiMonochromatorConfig Bandwidth="200" '
    string+=f'From="2300" Step="10" To="9000" Wavelength="{ex}0" />'
    string+='</FiWavelengthConfig.MonochromatorConfig></FiWavelengthConfig>'
    string+='</FluorescenceIntensityScanStrip.ExcitationMode>'
    string+='<FluorescenceIntensityScanStrip.ExcitationScanConfig><FiScanSettings '
    string+='Bandwidth="200" From="4400" Step="20" To="5000" />'
    string+='</FluorescenceIntensityScanStrip.ExcitationScanConfig>'
    string+='<FluorescenceIntensityScanStrip.GainSettings><FiScanGainManual Gain="100" '
    string+='Percent="100" /></FluorescenceIntensityScanStrip.GainSettings>'

    if idx == 0:
        string+='<FluorescenceIntensityScanStrip.GainWell><Well CartesianCoordinate="{x:Null}" Color="{x:Null}" ExperimentalGroup="{x:Null}" IdentifierGroupMember="{x:Null}" IdentifierReplicate="{x:Null}" IdentifierReplicates="{x:Null}" x:Name="__ReferenceID0" AlphanumericCoordinate="A1" Column="0" Grid="0" IdentifierGroup="None" IsFlagged="False" IsOut="False" IsSelected="True" Row="0" WellIndex="0"/></FluorescenceIntensityScanStrip.GainWell>'

    string+='<FluorescenceIntensityScanStrip.TopMirror><FiMirrorConfig '
    string+='EmissionEndWavelength="9000" EmissionStartWavelength="2800" '
    string+='ExcitationEndWavelength="9000" ExcitationStartWavelength="2300" '
    string+='IsAutomatic="False" IsUserDefinable="True" MirrorType="Automatic" '
    string+='Name="AUTOMATIC" /></FluorescenceIntensityScanStrip.TopMirror>'
    string+='<FluorescenceIntensityScanStrip.ZpositionTypeModel><FiZpositionTypeModel '
    string+='ZPositionType="Manual"><FiZpositionTypeModel.ZpositionModel>'
    string+=f'<FiZpositionManualModel Zposition="{zposition}" />'
    string+='</FiZpositionTypeModel.ZpositionModel></FiZpositionTypeModel>'
    string+='</FluorescenceIntensityScanStrip.ZpositionTypeModel></FluorescenceIntensityScanStrip>'
    
    # return string
    return string, id+1


def spark_end(id,
              numflashes=3,
              abs_nm=720,
              settletime=20
              ):
    
    string ='<AbsorbanceStrip SelectedInputData="{x:Null}" MeasurementsCount="1" '
    string+=f'MultipleReadsPerWell="False" NumberFlashes="{numflashes}" NumberOfMRWPoints="0" '
    string+='PathlengthCorrectionFactor="0.186" Reference="False" SelectedBorder="500" '
    string+='SelectedMultipleReadsPerWell="NotDefined" SelectedPathLengthCorrectionFactorType'
    string+='="Manual" SelectedPathLengthStatus="NotDefined" SelectedPattern="Square" '
    string+=f'SelectedSize="2" SettleTime="{settletime}" TestWavelength="9770" '
    string+=f'WavelengthMeasurement="{abs_nm}0" WavelengthReference="6200">'
    string+='<AbsorbanceStrip.DataLabels><DataLabel InternalSuffix="{x:Null}" '
    string+=f'x:Name="__ReferenceID{id+1}" Index="{id-1}" MeasureMode="SinglePoint" '
    string+=f'OutputName="Label {id}" Type="Measurement" Unit="OpticalDensity" />'
    string+='</AbsorbanceStrip.DataLabels></AbsorbanceStrip></PlateStrip>'
    string+='</InstrumentStrip><DataAnalysisStrip /><ExportStrip>'
    string+='<ExcelExportStrip EndTestSettings="{x:Null}" '
    string+='TemplateFilePathname="{x:Null}" TemplateSheetname="{x:Null}" '
    string+='AddToLastWorkbook="False"><ExcelExportStrip.DataLabels>'
    string+='<sco:ObservableCollection x:TypeArguments="IDataLabel" />'
    string+='</ExcelExportStrip.DataLabels></ExcelExportStrip>'
    string+='</ExportStrip></MethodStrip>'

    return string




