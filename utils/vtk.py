import vtk
from vtk.util import numpy_support

def render_mesh(objects, dim):
    actors = []

    for i, o in enumerate(objects):
        actors.append(create_actor(o['data'], dim, o['color'], o['opacity']))

    window = vtk.vtkRenderWindow()
    window.SetSize(500, 500)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)

    renderer = vtk.vtkRenderer()
    window.AddRenderer(renderer)

    for actor in actors:
        renderer.AddActor(actor)

    renderer.SetBackground(1.0, 1.0, 1.0)

    window.Render()
    interactor.Start()

def create_actor(data_matrix, dim, color='Yellow', opacity=1.0):
    # Convert input data
    data_string = data_matrix.tostring()

    # Import converted input data and set final dimensions
    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)

    dataImporter.SetDataExtent(0, dim-1, 0, dim-1, 0, dim-1)
    dataImporter.SetWholeExtent(0, dim-1, 0, dim-1, 0, dim-1)

    # Create mesh
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(dataImporter.GetOutputPort())
    dmc.Update()

    # Smooth mesh
    smoother = smooth_mesh(dmc, 20)

    mapper = vtk.vtkPolyDataMapper()
    mapper.ScalarVisibilityOff()
    mapper.SetInputConnection(smoother.GetOutputPort())

    rgbColor = get_color_rgb(color)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(rgbColor[0], rgbColor[1], rgbColor[2])

    return actor

def smooth_mesh(dmc, iterations=20):
    inputPoly = vtk.vtkPolyData()
    inputPoly.ShallowCopy(dmc.GetOutput())

    cleanPolyData = vtk.vtkCleanPolyData()
    cleanPolyData.SetInputData(inputPoly)
    cleanPolyData.Update()

    smooth_butterfly = vtk.vtkButterflySubdivisionFilter()
    smooth_butterfly.SetNumberOfSubdivisions(0)
    smooth_butterfly.SetInputConnection(cleanPolyData.GetOutputPort())
    smooth_butterfly.Update()

    upsampledInputPoly = vtk.vtkPolyData()
    upsampledInputPoly.DeepCopy(smooth_butterfly.GetOutput())

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(upsampledInputPoly)
    decimate.SetTargetReduction(0.0)
    decimate.PreserveTopologyOn()
    decimate.Update()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(decimate.GetOutputPort())
    smoother.SetNumberOfIterations(20) #
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.FlipNormalsOn()

    return normals

def get_color_rgb(name):
    namedColors = vtk.vtkNamedColors()
    
    return namedColors.GetColor3d(name)
