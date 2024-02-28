-- needed to speed up queries; do at end to speed up loading
alter table drugsFromCVAROD add index (event);
alter table drugsFromCVAROD add index (drug);
alter table diseasesFromCVAROD add index (event);
alter table diseasesFromCVAROD add index (indi);
alter table demoFromCVAROD add index (event);
