import numpy as np
import vtkmodules.all as vtk


def arr3d2vtr(data: np.ndarray, filename: str):
    if len(np.shape(data)) != 3:
        raise TypeError('The input data must be 3d!')
    if not filename.endswith('.vtr'):
        filename += '.vtr'
    lenx = len(data[:, 0, 0])
    leny = len(data[0, :, 0])
    lenz = len(data[0, 0, :])
    x = list(range(lenx+1))
    y = list(range(leny+1))
    z = list(range(lenz+1))
    xCoords = vtk.vtkFloatArray()
    for i in x:
        xCoords.InsertNextValue(i)

    yCoords = vtk.vtkFloatArray()
    for i in y:
        yCoords.InsertNextValue(i)

    zCoords = vtk.vtkFloatArray()
    for i in z:
        zCoords.InsertNextValue(i)

    rgrid = vtk.vtkRectilinearGrid()
    rgrid.SetDimensions(len(x), len(y), len(z))
    rgrid.SetXCoordinates(xCoords)
    rgrid.SetYCoordinates(yCoords)
    rgrid.SetZCoordinates(zCoords)

    scalars = vtk.vtkDoubleArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetNumberOfTuples(lenx * leny * lenz)
    for k in range(0, lenz):
        kOffset = k * lenx * leny
        for j in range(0, leny):
            jOffset = j * lenx
            for i in range(0, lenx):
                s = data[i, j, k]
                offset = i + jOffset + kOffset
                scalars.InsertTuple1(offset, s)
    rgrid.GetCellData().SetScalars(scalars)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(rgrid)
    writer.Write()


if __name__ == "__main__":
    data = np.random.rand(4, 5, 6)
    arr3d2vtr(data, 'test.vtr')
