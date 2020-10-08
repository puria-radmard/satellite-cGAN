class LSTNAggregationOperation(AggregationOperation):
  band_required = "LST"

  def operation(self, array_dict):

    master_values = np.concatenate([pair[0].reshape(-1) for pair in array_dict.values()])
    raster_meta = list(array_dict.values())[0][-1]
    mean = np.nanmean(master_values)
    std = np.nanstd(master_values)

    for LST_path, (LST_map, raster_meta) in array_dict.items():
      LSTN_map = LST_map.copy()
      nonnan = np.where(~np.isnan(LSTN_map))
      LSTN_map[nonnan] = (LST_map[nonnan] - mean)/std
      UHI_path = self.replace_band_name(LST_path, "LSTN")
      save_calculated_raster(
          raster_meta,
          path=UHI_path,
          image=LSTN_map[np.newaxis,:]
      )


class UHIAggregationOperation(AggregationOperation):
  band_required = "LSTN"

  def operation(self, array_dict):

    master_values = np.concatenate([pair[0].reshape(-1) for pair in array_dict.values()])
    raster_meta = list(array_dict.values())[0][-1]
    mean = np.nanmean(master_values)
    std = np.nanstd(master_values)
    # Should be close to 1 given this is LSTN
    thres = mean + std

    for LSTN_path, (LSTN_map, raster_meta) in array_dict.items():
      UHI_map = LSTN_map.copy()
      nonnan = np.where(~np.isnan(UHI_map))
      UHI_map[nonnan] = (UHI_map[nonnan]>thres).astype(int)
      UHI_path = self.replace_band_name(LSTN_path, "UHI")
      save_calculated_raster(
          raster_meta,
          path=UHI_path,
          image=UHI_map[np.newaxis,:]
      )